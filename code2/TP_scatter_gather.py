import torch
import torch.nn as nn
import torch.distributed as dist
import os

def main():
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = f'cuda:{local_rank}'
    dist.init_process_group(backend='nccl')
    
    tensor_size = 2

    output_tensor = torch.zeros(tensor_size, device=device)
    
    if dist.get_rank() == 0:
        t_ones       = torch.ones(tensor_size, device=device)
        t_fives      = torch.ones(tensor_size, device=device) * 5
        scatter_list = [t_ones, t_fives]
    else:
        scatter_list = None

	# scatter_list  - list of tensors to be scattered
	# output_tensor - where the received tensor will be stored in each node
    dist.scatter(output_tensor, scatter_list, src=0)

    print(f'Local rank {local_rank} tensor received from scatter: {output_tensor}')

    g_in_tensor = output_tensor + 1

    if dist.get_rank() == 0:
        t_ones1     = torch.ones(tensor_size, device=device)
        t_ones2     = torch.ones(tensor_size, device=device)
        gather_list = [t_ones1, t_ones2]
    else:
        gather_list = None

    # g_in_tensor - tensor to gather from all nodes
    # gather_list - rank 0 list where the gathered tensors will be stored
    dist.gather(g_in_tensor, gather_list, dst=0)
    
    if dist.get_rank() == 0:
        print(f'Gathered list of tensors: {gather_list}')

if __name__ == "__main__":
    main()