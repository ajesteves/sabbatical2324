This folder contains a collection of scripts that support the report "Distributed Training of Deep Neural Networks". The code is organized in the following way:

* `collective_ops.py`: explore collective communication operations.
* `cifar10_ddp_02.py`: distributed data parallelism (DDP) training on multiple GPUs.
* `parameter_server.py`: parameter server implementation.
* `FSDP2_pytorch_tutorial\`: apply PyTorch fully sharded data parallelism (FSDP2) to a transformer model.
* `pipelining_tutorial.py`: apply pipeline parallelism to a simple transformer decoder-only model.
* `TP_scatter_gather.py`: explore scatter and gather collective communication operations.
* `TP_linear2_v1.py`: apply row-wise tensor parallelism to a Linear layer.
* `TP_linear2_v2.py`: apply column-wise and row-wise parallelism to Linear layers.
* `nanoMoE/`: train a MoE model based on Andrej Karphathy nanoGPT.
* `deepspeedMoE/`: apply DeepSpeed-MoE to a simple CIFAR-10 CNN model.
