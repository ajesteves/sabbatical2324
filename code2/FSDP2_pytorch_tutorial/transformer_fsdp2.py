import argparse
import os

import torch
from   checkpoint             import Checkpointer
from   model                  import ModelArgs, Transformer
from   torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from   utils                  import inspect_mixed_precision, inspect_model

# ....................................................................
def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """
    Verify that we have at least 2 GPUs to run this distributed example.
    """
    has_gpu   = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

# ....................................................................
def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """
    Set prefetch AllGather operations over model layers in forward pass.
    """
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)

# ....................................................................
def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """
    Set prefetch AllGather operations over model layers in backward pass.
    """
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)

# ....................................................................
def main(args):
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} GPUs to run this example. Exiting.")
        exit()

    rank = int(os.environ["LOCAL_RANK"])
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator()
        device      = torch.device(f"{device_type}:{rank}")
        torch.accelerator.device_index(rank)
        print(f"Running rank {rank} on device {device}")
    else:
        device = torch.device("cpu")
        print(f"Running on device {device}")

    # Initialize the default process group and communication backend.

    backend = torch.distributed.get_default_backend_for_device(device)
    torch.distributed.init_process_group(backend=backend, device_id=device)

    # Configure a Transformer model with 10 layers, 4 attention heads, 
    # a maximum sequence length of 64, no dropout, and a vocabulary size of 1024.

    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len    = 64
    model_args = ModelArgs(
        n_layers    = 10,
        n_heads     = 4,
        vocab_size  = vocab_size,
        max_seq_len = seq_len,
        dropout_p   = 0,
    )

    # Instantiate the model on the meta device to avoid unnecessary memory usage, 
    # since FSDP2 will take care of moving parameters to the correct device and shard them.

    with torch.device("meta"):
        model = Transformer(model_args)

    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype  = torch.bfloat16,
            reduce_dtype = torch.float32,
        )

    # Wrap each layer with FSDP2 and then the entire model. 
    # This allows for more fine-grained control over sharding and prefetching.

    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    # Print the model to verify that it is correctly wrapped with FSDP2. 
    inspect_model(model)

    # Optionally set up explicit prefetching for the forward and backward passes.
    # With prefetching, FSDP modules should explicitly prefetch AllGathers in forward. 
    # The prefetching runs after this module’s AllGather copy-out.

    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model,  num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    # If a checkpoint exists, load the model and optimizer state from the checkpoint.

    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.last_training_time is None:
        model.to_empty(device=device)
        model.reset_parameters()
    else:
        checkpointer.load_model(model)
    
    # If mixed precision is enabled, inspect the model to verify that 
    # weights are in the correct data type.

    if args.mixed_precision:
        inspect_mixed_precision(model)

    # Construct the optimizer based on the model parameters. If a checkpoint was loaded,
    # load the optimizer state from the checkpoint as well.

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    if checkpointer.last_training_time is not None:
        checkpointer.load_optim(model, optim)

    # Run a simple training loop for 10 iterations, where in each iteration we:
    # 1. Optionally unshard the model if explicit prefetching is enabled
    # 2. Generate random input data
    # 3. Compute the loss and perform the backward pass
    # 4. Clip the gradients and update the model weights using the optimizer 
    # 5. Reset the optimizer gradients

    for _ in range(10):
        if args.explicit_prefetching:
            model.unshard()
        x    = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

    # Save the model and optimizer state to a checkpoint, 
    # and then clean up the process group.

    checkpointer.save(model, optim)
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision",      action="store_true", default=False)
    parser.add_argument("--dcp-api",              action="store_true", default=False)
    args = parser.parse_args()
    
    main(args)
