"""
This script applies Tensor Parallel(TP) to a simple model in a
Megetron-LM SPMD style. We show an end-to-end working flow including forward,
backward, and optimization.

The model consists of two 'Linear' layers with an element-wise 'ReLU'
in between.

The basic idea is that we apply column-wise parallelization to the first linear layer,
and also apply row-wise parallelization to the second linear layer so that we only need
one AllReduce in the end of the second linear layer.

We speed up the model training by avoiding communications between
the two Linear layers.

To parallelize an 'nn.module', we need to specify what parallel style we want
to use and our 'parallelize_module' API will parse and parallelize the modules
based on the given 'ParallelStyle'. We are using this PyTorch native Tensor
Parallelism APIs to show how it works.
"""

import os
import sys
import logging
import time
import torch
import torch.nn as nn
from   torch.distributed.tensor.parallel import (
       parallelize_module,
       ColwiseParallel,
       RowwiseParallel,
       )
from   torch.distributed._tensor.device_mesh import init_device_mesh

# ---------------------------
logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

# ---------------------------
def get_logger():
    return logging.getLogger(__name__)

# ---------------------------
def rank_log(_rank, logger, msg):
    """
    Helper function to log only on global rank 0.
    """
    if _rank == 0:
        logger.info(f" {msg}")

# ---------------------------
def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """
    Verify that we have at least 2 GPUs to run this example.
    """
    has_cuda  = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    return has_cuda and gpu_count >= min_gpus

# ---- GPU check ------------
_min_gpu_count = 2

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()

# ---------------------------
class SimpleModel(nn.Module):
    """
    Simple model with two Linear layers and two ReLU activations.
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1  = nn.Linear(100, 50)
        self.relu1    = nn.ReLU()
        self.linear2  = nn.Linear(50, 20)
        self.relu2    = nn.ReLU()

    def forward(self, x):
        return self.relu2(self.linear2(self.relu1(self.linear1(x))))


"""
Main body of the demo of a basic version of tensor parallel by using
PyTorch native APIs.
"""
logger = get_logger()

# Create a device mesh based on the given world_size.
_world_size = int(os.environ["WORLD_SIZE"])

device_mesh = init_device_mesh(
    device_type="cuda", 
    mesh_shape=(_world_size,)
    )
_rank       = device_mesh.get_rank()

print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require an even number of GPUs, but got {_world_size} gpus"

rank_log(_rank, logger, f"Device mesh created: {device_mesh=}")

# Create the model and move it to GPU.
# init_device_mesh has already assigned IDs to GPUs.
tp_model = SimpleModel().to("cuda")

# Create an optimizer for the parallelized module.
lr        = 0.25
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr, foreach=True)

# Custom parallelization plan for the model: column-wise parallel for the 
# first linear layer and row-wise parallel for the second linear layer.
tp_model = parallelize_module(
    module           = tp_model,
    device_mesh      = device_mesh,
    parallelize_plan = {
        "linear1": ColwiseParallel(),
        "linear2": RowwiseParallel(),
    },
)
# Perform a number of iterations of forward/backward
# and optimizations for the sharded module.
num_iters = 10000
rank_log(_rank, logger, "Tensor Parallel training starting...")

if _rank == 0:
        ts = time.time()

for id in range(num_iters):
    # For TP, input needs to be the same across all ranks.
    # Setting the random seed is to mimic the behavior of a dataloader.
    torch.manual_seed(id)
    optimizer.zero_grad()
    inp    = torch.rand(32, 100, device="cuda")
    output = tp_model(inp)
    output.sum().backward()
    optimizer.step()
    if id % 100 == 0:
        rank_log(_rank, logger, f"Tensor parallel iteration {id} completed")

rank_log(_rank, logger, "Tensor parallel training completed!")
if _rank == 0:
    te = time.time()
    print(f"Execution time for 10k iterations: {te-ts} s")