import os
import time
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.distributed   as dist

# Linear layer with row-wise tensor parallelism.
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features        = in_features
        self.out_features       = out_features

        self.rank               = dist.get_rank()
        self.world_size         = dist.get_world_size()
        self.device             = f'cuda:{self.rank}'

		# row-wise tensor parallelism applied to Linear layer
        self.local_in_features  = in_features // self.world_size
        self.local_out_features = out_features

        self.linear = nn.Linear(self.local_in_features, self.local_out_features)
    
    def forward(self, x, batch_size):
        
        # input shape = [batch_size , local_in_features]
        local_input = torch.zeros(batch_size, self.local_in_features, device=self.device)

		# scatter a parcel of input with size [batch_size,local_in_features/world_size]
		# to every GPU (input is split column-wise)
        dist.scatter(
           local_input,
           list(x.chunk(self.world_size, dim=1)) if self.rank == 0 else None, 
           src=0
        )

		# Pass input through the Linear layer
        output1 = self.linear(local_input)

		# AllReduce partial Linear layers' outputs from all GPUs using sum
		# All GPus will have the same output of the Linear layer
        dist.all_reduce(output1, op=dist.ReduceOp.SUM)

        return output1

# Simple model with Linear, ReLU, Linear, ReLU layers.
class Model(nn.Module):
    def __init__(self,inf, hidf, outf, device) -> None:
        super().__init__()
        self.linear1 = Linear(inf, hidf).to(device)
        self.linear2 = Linear(hidf, outf).to(device)

    def forward(self, x, bs):
        x = F.relu(self.linear1(x, bs))
        return F.relu(self.linear2(x, bs))

def main():
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = f'cuda:{local_rank}'
    dist.init_process_group(backend='nccl')

    batch_size = 32

    model = Model(inf=100, hidf=50, outf=20, device=device)

    # Create an optimizer for the parallelized module
    lr        = 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

    if dist.get_rank() == 0:
        ts = time.time()

    for id in range(10000):

        optimizer.zero_grad()
        if dist.get_rank() == 0:
            input_tensor = torch.randn(batch_size, 100, device=device)
        else:
            input_tensor = None

        output = model(input_tensor, batch_size)
        output.sum().backward()
        optimizer.step()

        if dist.get_rank() == 0 and id % 100 == 0:
            print(f"Iteration {id} completed.")
    
    if dist.get_rank() == 0:
        te = time.time()
        print(f"Execution time for 10k iterations: {te-ts} s")
        print(f'{output}\n {output.shape}')

if __name__ == "__main__":
    main()