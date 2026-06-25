import argparse
import os

import deepspeed
import torch
import torchvision
import torch.nn              as     nn
import torch.nn.functional   as     F
from   torchvision           import transforms
from   deepspeed.accelerator import get_accelerator
from   deepspeed.moe.utils   import split_params_into_different_moe_groups_for_optimizer

def add_argument():
    """
    Parse the command line arguments for the CIFAR10 training script.
    """
    parser = argparse.ArgumentParser(description="CIFAR10")

    # Arguments related to training
    parser.add_argument(
        "-e",
        "--epochs",
        default = 30,
        type    = int,
        help    = "number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--local_rank",
        type    = int,
        default = -1,
        help    = "local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--log-interval",
        type    = int,
        default = 2000,
        help    = "interval between successive logs (default: 2000)",
    )

    # Argument for mixed precision training
    parser.add_argument(
        "--dtype",
        default = "fp16",
        type    = str,
        choices = ["bf16", "fp16", "fp32"],
        help    = "Datatype used for training",
    )

    # Arguments for ZeRO Optimization
    parser.add_argument(
        "--stage",
        default = 0,
        type    = int,
        choices = [0, 1, 2, 3],
        help    = "ZeRO optimization stage",
    )

    # Arguments related to Mixture of Experts
    parser.add_argument(
        "--moe",
        default = False,
        action  = "store_true",
        help    = "use DeepSpeed mixture of experts",
    )
    parser.add_argument(
        "--ep-world-size",
        default = 1,
        type    = int, 
        help    = "expert parallelism degree"
    )
    parser.add_argument(
        "--num-experts",
        type    = int,
        nargs   = "+",
        default = [1, ],
        help="number of experts layers (as a list), more than 1 value mean PR-MOE architecture.",
    )
    parser.add_argument(
        "--mlp-type",
        type    = str,
        default = "standard",
        help    = "Only applicable when num-experts > 1, accepts [standard, residual]",
    )
    parser.add_argument(
        "--top-k", 
        default = 1,
        type    = int,
        help    = "MoE gating top-k (only top-1 and top-2 are supported)"
    )
    parser.add_argument(
        "--min-capacity",
        default = 0,
        type    = int,
        help    = "MoE minimum capacity of an expert, regardless of the capacity factor",
    )
    parser.add_argument(
        "--noisy-gate-policy",
        default = None,
        type    = str,
        help    = "MoE noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter",
    )
    parser.add_argument(
        "--moe-param-group",
        default = False,
        action  = "store_true",
        help    = "Create separate MoE weight groups, required when using ZeRO with MoE",
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def create_moe_param_groups(model):
    """
    Create a separate weight group for each MoE expert.
    """
    weights = {"params": [p for p in model.parameters()], "name": "parameters"}
    return split_params_into_different_moe_groups_for_optimizer(weights)


def get_ds_config(args):
    """
    Get the DeepSpeed configuration dictionary.
    """
    ds_config = {
        "train_batch_size":     16,
        "steps_per_print":      64,
        "optimizer": {
            "type":             "Adam",
            "params": {
                "lr":           0.001,
                "betas":        [0.8, 0.999],
                "eps":          1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr":    0,
                "warmup_max_lr":    0.001,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping":  1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": args.dtype == "bf16"},
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale":            0,
            "loss_scale_window":     500,
            "hysteresis":            2,
            "min_loss_scale":        1,
            "initial_scale_power":   15,
        },
        "wall_clock_breakdown":      False,
        "zero_optimization": {
            "stage":                 args.stage,
            "allgather_partitions":  True,
            "reduce_scatter":        True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size":    50000000,
            "overlap_comm":          True,
            "contiguous_gradients":  True,
            "cpu_offload":           False,
        },
    }
    return ds_config


class CNN(nn.Module):
    """
    A simple convolutional neural network for CIFAR10 classification.
    """
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.moe   = args.moe
        if self.moe:
            # An MoE expert is a Linear layer with 84 input and 84 output features
            fc3 = nn.Linear(84, 84)
            self.moe_layer_list = []
            # Create 'n_e' MoE experts, each equal to 'fc3'
            for n_e in args.num_experts:
                self.moe_layer_list.append(
                    deepspeed.moe.layer.MoE(
                        hidden_size       = 84,
                        expert            = fc3,
                        num_experts       = n_e,
                        ep_size           = args.ep_world_size,
                        use_residual      = args.mlp_type == "residual",
                        k                 = args.top_k,
                        min_capacity      = args.min_capacity,
                        noisy_gate_policy = args.noisy_gate_policy,
                    )
                )
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
            self.fc4            = nn.Linear(84, 10)
        else:
            self.fc3            = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.moe:
            for layer in self.moe_layer_list:
                x, _, _ = layer(x)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x


def test(model_engine, testset, local_device, target_dtype, test_batch_size=4):
    """
    Evaluate the model on the test set.

    Args:
        model_engine:    the DeepSpeed engine (deepspeed.runtime.engine.DeepSpeedEngine).
        testset:         the test set (torch.utils.data.Dataset).
        local_device:    the local device name (str).
        target_dtype:    the target datatype for the test data (torch.dtype).
        test_batch_size: the test batch size (int).

    """
    # The 10 classes for CIFAR10.
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Define the test DataLoader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size  = test_batch_size,
        shuffle     = False, 
        num_workers = 2,
    )

    # To calculate the total accuracy
    correct, total = 0, 0
    
    # To calculate the accuracy per class
    class_correct = list(0.0 for i in range(10))
    class_total   = list(0.0 for i in range(10))

    # Test loop ......................................................
    
    model_engine.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if target_dtype != None:
                images = images.to(target_dtype)
            outputs = model_engine(images.to(local_device))
            _, predicted = torch.max(outputs.data, 1)
            # Count the total predictions and the correct predictions
            total   += labels.size(0)
            correct += (predicted == labels.to(local_device)).sum().item()

            # Calculate the accuracy for each class
            batch_correct = (predicted == labels.to(local_device)).squeeze()
            for i in range(test_batch_size):
                label                 = labels[i]
                class_correct[label] += batch_correct[i].item()
                class_total[label]   += 1

    if model_engine.local_rank == 0:
        print(
            f"Accuracy of the network on the {total} test images: {100 * correct / total : .0f} %"
        )

        # Print the accuracy of all classes
        for i in range(10):
            print(
                f"Accuracy of {classes[i] : >5s} : {100 * class_correct[i] / class_total[i] : 2.0f} %"
            )


def main(args):
    # Initialize the DeepSpeed distributed backend
    deepspeed.init_distributed()
    _local_rank = int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(_local_rank)

    #......................................................................
    # Step 1. Data preparation
    #
    # The output of torchvision datasets are PILImage images in [0, 1] range.
    # Transform the images to Tensors and normalized to the [-1, 1] range.
    #......................................................................
    transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    if torch.distributed.get_rank() != 0:
        # It might be necessary to download cifar10 dataset.
        # Rank 0 will perform the downloading.
        torch.distributed.barrier()

    # Download/load cifar10 training set in rank 0
    trainset = torchvision.datasets.CIFAR10(
        root      = "./data", 
        train     = True, 
        download  = True, 
        transform = transform,
    )
    
    testset = torchvision.datasets.CIFAR10(
        root      = "./data", 
        train     = False,
        download  = True,
        transform = transform
    )

    if torch.distributed.get_rank() == 0:
        # Cifar10 training data is downloaded.
        # Signal the other ranks that they can proceed.
        torch.distributed.barrier()

    #......................................................................
    # Step 2. Wrap the model with DeepSpeed
    #
    # First, we define a convolution neural network.
    # Then, we define the DeepSpeed configuration dictionary 
    # and use it to initialize the DeepSpeed engine.
    #......................................................................

    # Instantiate the convolution neural network
    model = CNN(args)

    # Get the list of weights that require gradients
    weights = filter(lambda p: p.requires_grad, model.parameters())

    # Create a separate weight group for each MoE expert
    if args.moe_param_group:
        weights = create_moe_param_groups(model)

    # Initialize DeepSpeed to use the following features:
    #   1) a distributed model
    #   2) a distributed data loader
    #   3) a DeepSpeed optimizer
    
    ds_config = get_ds_config(args)
    
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args             = args,
        model            = model,
        model_parameters = weights,
        training_data    = trainset,
        config           = ds_config,
    )

    # Get the local device name (str) and the local rank (int)
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank   = model_engine.local_rank

    # For float32, 'target_dtype' will be 'None', meaning that no datatype 
    # conversion is necessary
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    # Select the cross-entropy loss function
    criterion = nn.CrossEntropyLoss()

    #......................................................................
    # Step 3. Train the model
    #
    # We simply have to iterate over the DataLoader, and apply the images 
    # at the model input and run the optimization.
    # DeepSpeed handles the distributed training aspects for us.
    #......................................................................

    # Iterate over the dataset multiple times
    for epoch in range(args.epochs):  
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # The loaded minibatch of images and labels is a list with two tensors
            inputs, labels = data[0].to(local_device), data[1].to(local_device)

            # Convert images to 'target_dtype' if necessary
            if target_dtype != None:
                inputs = inputs.to(target_dtype)

            # Run the forward pass
            outputs = model_engine(inputs)

            # Compute the loss
            loss    = criterion(outputs, labels)

            # Calculate the loss gradients
            model_engine.backward(loss)

            # Update the model weights
            model_engine.step()

            # Print the training statistics
            running_loss += loss.item()
            if local_rank == 0 and i % args.log_interval == (args.log_interval - 1):
                # Print statistics every 'log_interval' minibatches
                print(
                    f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .3f}"
                )
                running_loss = 0.0

    print("[INFO] Finished training")

    #......................................................................
    # Step 4. Evaluate the model on the test set
    #......................................................................
    test(model_engine, testset, local_device, target_dtype)


if __name__ == "__main__":
    args = add_argument()
    main(args)
