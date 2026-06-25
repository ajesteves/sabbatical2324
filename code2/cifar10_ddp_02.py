# --------------------------------------------------------------------
'''
Distributed Data Parallel (DDP) training on multiple GPUs.

Source:
https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-3-distributed-data-parallel-d26e93f17c62

DDP is more intrusive into your code than DP, so we need to modify multiple parts of the single-GPU code.

1. DDP initialization. 

Rank is the unique ID of our GPU, and world_size is the total processes, which is the number of GPUs 
since each process controls one GPU. 
The init_process_group currently supports three types of backends: gloo, nccl, and mpi.
The nccl is required if we want to build with CUDA.
'''
# --------------------------------------------------------------------

import os
import time
import torch
import torchmetrics
from   pathlib                      import Path
import torch.multiprocessing        as     mp
import torch.nn                     as     nn
from   torch.nn.parallel            import DistributedDataParallel as DDP
from   torch.utils.data.distributed import DistributedSampler

from   torch.distributed            import init_process_group, destroy_process_group
from   torch.utils.data             import Dataset, DataLoader
from   torchvision.datasets         import CIFAR10
from   torchvision.models           import resnet34
from   torchvision.transforms       import transforms
import torch.optim                  as     optim
from   torch                        import Tensor
from   typing                       import Iterator, Tuple

# --------------------------------------------------------------------
'''
All trained models are saved at ?./models?, and the CIFAR10 dataset is saved at ?./data?. 
These hyperparameters will stay the same during multi-GPU training.
'''

def configure() -> dict:
    """
    Setup data directory, model directory, and training hyperparameters.
    """
    data_root      = Path("data")
    trained_models = Path("models")

    if not data_root.exists():
        data_root.mkdir()

    if not trained_models.exists():
        trained_models.mkdir()

    config = dict(
        data_root      = data_root,
        trained_models = trained_models,
        total_epochs   = 15,
        batch_size     = 128,
        lr             = 0.1,  # learning rate
        momentum       = 0.9,
        lr_step_size   = 5,
        save_every     = 3,
    )

    return config

# --------------------------------------------------------------------
'''
Torchvision ResNets are defined for ImageNet, which has a higher resolution than CIFAR10, 
so we replace the stem stage with a smaller kernel_size (7 to 3) and remove the maxpooling layer.
'''

def cifar_model() -> nn.Module:
    model         = resnet34(num_classes=10)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# --------------------------------------------------------------------
'''
CIFAR10 dataset.
'''

def cifar_dataset(data_root: Path) -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.49139968, 0.48215827, 0.44653124),
                std  = (0.24703233, 0.24348505, 0.26158768),
            ),
        ]
    )

    trainset = CIFAR10(root=data_root, train=True,  transform=transform, download=True)
    testset  = CIFAR10(root=data_root, train=False, transform=transform, download=True)

    return trainset, testset

# --------------------------------------------------------------------
'''
Sequential trainer class.

We use torchmetrics to compute the classification accuracy since it 
supports distributed scenarios. We will verify its correctness when using DDP. 
Note "torchmetrics.Accuracy" contain parameters. So, it has to be on GPU.
The code is pretty straightforward as "_run_batch" takes care of each batch and 
"_run_epoch" takes care of each epoch.
The "lr_scheduler" decreases the learning rate by 5 (???) every 5 epochs.
'''

class TrainerSingle:
    def __init__(
        self,
        gpu_id:      int,
        model:       nn.Module,
        trainloader: DataLoader,
        testloader:  DataLoader,
    	):
        self.gpu_id      = gpu_id

        self.config      = configure()
        self.model       = model.to(self.gpu_id)
        self.trainloader = trainloader
        self.testloader  = testloader
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = optim.SGD(
            self.model.parameters(),
            lr       = self.config["lr"],
            momentum = self.config["momentum"],
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, self.config["lr_step_size"]
        )
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=10, average="micro"
        ).to(self.gpu_id)

        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=10, average="micro"
        ).to(self.gpu_id)

    def _run_batch(self, img: Tensor, target: Tensor) -> float:
        self.optimizer.zero_grad()

        out  = self.model(img)
        loss = self.criterion(out, target)
        loss.backward()
        self.optimizer.step()

        self.train_acc.update(out, target)
        return loss.item()

    def _run_epoch(self, epoch: int):

        loss = 0.0
        for img, target in self.trainloader:
            img        = img.to(self.gpu_id)
            target     = target.to(self.gpu_id)
            loss_batch = self._run_batch(img, target)
            loss      += loss_batch
        self.lr_scheduler.step()

        print(
            f"{'-' * 90}\n[GPU{self.gpu_id}] Epoch {epoch:2d} | Batchsize: {self.config['batch_size']} | Steps: {len(self.trainloader)} | LR: {self.optimizer.param_groups[0]['lr']:.4f} | Loss: {loss / len(self.trainloader):.4f} | Acc: {100 * self.train_acc.compute().item():.2f}%",
            flush=True,
        )

        self.train_acc.reset()

    def _save_checkpoint(self, epoch: int):
        ckp        = self.model.state_dict()
        model_path = self.config["trained_models"] / f"CIFAR10_single_epoch{epoch+1}.pt"
        torch.save(ckp, model_path)

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.config["save_every"] == 0:
                self._save_checkpoint(epoch)
        # save in last epoch
        self._save_checkpoint(max_epochs - 1)

    def test(self, final_model_path: str):
        self.model.load_state_dict(torch.load(final_model_path))
        self.model.eval()
        with torch.no_grad():
            for img, target in self.testloader:
                img    = img.to(self.gpu_id)
                target = target.to(self.gpu_id)
                out    = self.model(img)
                self.valid_acc.update(out, target)
        print(
            f"[GPU{self.gpu_id}] Test Acc: {100 * self.valid_acc.compute().item():.4f}%"
        )

# --------------------------------------------------------------------
'''
Train and test DataLoaders.

The DataLoader determines how we load the data into batches during training, 
evaluation, and testing time. In DDP, the DistributedSampler ensures each device gets 
a non-overlapping input batch.
'''
def cifar_dataloader_ddp(
    trainset: Dataset,
    testset:  Dataset,
    bs:       int,
	) -> Tuple[DataLoader, DataLoader, DistributedSampler]:

    sampler_train = DistributedSampler(trainset, shuffle=True)
    trainloader   = DataLoader(
        trainset, 
        batch_size  = bs, 
        shuffle     = False, 
        sampler     = sampler_train, 
        num_workers = 8,
    )

    sampler_test = DistributedSampler(testset, shuffle=True)
    testloader   = DataLoader(
        testset,
        batch_size  = bs,
        shuffle     = False,
        sampler     = sampler_test,
        num_workers = 8,
    )

    return trainloader, testloader, sampler_train

# --------------------------------------------------------------------
'''
Trainer class for DDP.

TrainerDDP takes care of the training and testing process.
TrainerDDP inherits "TrainerSingle" so that we can better visualize what has changed 
from the single-GPU example.

In the DataLoader, we set "shuffle=False" due to the "DistributedSampler" taking care of suffling.

To shuffle the dataset while using DistributedSampler, we have to call the 
"DistributedSampler.set_epoch" method. We will explore whether this "set_epoch" shuffles intra- or 
cross- GPU in the experiment section.

In distributed mode, calling the "set_epoch()"" method at the beginning of each epoch before 
creating the DataLoader iterator is necessary to make shuffling work properly across multiple 
epochs. Otherwise, the same ordering will be always used.

We use "torchmetrics" to compute the classification accuracy due to its support of distributed 
scenarios. We will manually compute the accuracy and verify its correctness in the Experiment section.
'''

# Each process will launch a copy of this class
class TrainerDDP(TrainerSingle):
    def __init__(
        self,
        gpu_id:        int,
        model:         nn.Module,
        trainloader:   DataLoader,
        testloader:    DataLoader,
        sampler_train: DistributedSampler,
    	) -> None:
        super().__init__(gpu_id, model, trainloader, testloader)

        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.model         = DDP(self.model, device_ids=[gpu_id])
        self.sampler_train = sampler_train

    def _save_checkpoint(self, epoch: int):
        ckp        = self.model.state_dict()
        model_path = self.config["trained_models"] / f"CIFAR10_ddp_epoch{epoch+1}.pt"
        torch.save(ckp, model_path)

    def train(self, max_epochs: int):
        self.model.train()

        for epoch in range(max_epochs):
            self.sampler_train.set_epoch(epoch)

            self._run_epoch(epoch)

            # only save once on master gpu
            if self.gpu_id == 0 and epoch % self.config["save_every"] == 0:
                self._save_checkpoint(epoch)

        # save on the last epoch
        self._save_checkpoint(max_epochs - 1)

    def test(self, final_model_path: str):
        self.model.load_state_dict(
            torch.load(final_model_path, map_location="cpu")
        )
        self.model.eval()

        with torch.no_grad():
            for img, target in self.testloader:
                img    = img.to(self.gpu_id)
                target = target.to(self.gpu_id)
                out    = self.model(img)
                self.valid_acc.update(out, target)
        print(
            f"[GPU{self.gpu_id}] Test Acc: {100 * self.valid_acc.compute().item():.4f}%"
        )

# --------------------------------------------------------------------

# Each process control a single gpu/cpu
def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size) # "gloo" | "nccl"

# --------------------------------------------------------------------
'''
Main function.

We need an extra line of code that sets up the distributed scenario and another line of code 
that cleans up all processes after training.
'''
def main_ddp(
    rank:             int,
    world_size:       int,
    final_model_path: str,
	):

    ddp_setup(rank, world_size)  # initialize DDP 

    config = configure()

    train_dataset, test_dataset = cifar_dataset(config["data_root"])
    train_dataloader, test_dataloader, train_sampler = cifar_dataloader_ddp(
        train_dataset, 
        test_dataset, 
        config["batch_size"]
    )
    model   = cifar_model()

    trainer = TrainerDDP(
        gpu_id        = rank,
        model         = model,
        trainloader   = train_dataloader,
        testloader    = test_dataloader,
        sampler_train = train_sampler,
    )
    trainer.train(config["total_epochs"])
    trainer.test(final_model_path)

    destroy_process_group()  # kill processes

# --------------------------------------------------------------------
'''
Experiments.

Disable shuffling: To disable shuffling it is necessary to comment out the line
"self.sampler_train.set_epoch(epoch)" in the "TrainerDDP.train" method. 
In general, training with shuffling leads to higher accuracy.
'''

if __name__ == "__main__":
    world_size       = torch.cuda.device_count()
    final_model_path = Path("./models/CIFAR10_ddp_epoch15.pt")
    start_time = time.time()
    mp.spawn(
        main_ddp,
        args   = (world_size, final_model_path),
        nprocs = world_size, # number of processes - # gpus
    )
    end_time = time.time()
    texec_sec = end_time - start_time
    texec_min = int(texec_sec/60)
    texec_sec = int(texec_sec - texec_min * 60)

    print(f'Execution time: {texec_min}m {texec_sec}s')
