# ==================================================
# Brodcast

import os
import torch
import torch.distributed as ptd

def init_process():
    grp = ptd.init_process_group(backend='nccl')
    torch.cuda.set_device(ptd.get_rank())
    return grp

def example_broadcast():
    if ptd.get_rank() == 0:
        tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
    else:
        tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f"Tensor before broadcast on rank {ptd.get_rank()}: {tensor}")
    ptd.broadcast(tensor, src=0)
    print(f"Tensor after broadcast on rank {ptd.get_rank()}: {tensor}")

if __name__== '__main__':
    init_process()
    example_broadcast()

'''
Tensor before broadcast on rank 1: tensor([0., 0., 0., 0., 0.], device='cuda:1')
Tensor before broadcast on rank 0: tensor([1., 2., 3., 4., 5.], device='cuda:0')
Tensor after broadcast on rank 0:  tensor([1., 2., 3., 4., 5.], device='cuda:0')
Tensor after broadcast on rank 1:  tensor([1., 2., 3., 4., 5.], device='cuda:1')
'''
# ==================================================

def example_AllReduce():
    tensor = torch.tensor([ptd.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f"Tensor before AllReduce on rank {ptd.get_rank()}: {tensor}")
    ptd.all_reduce(tensor, op=ptd.ReduceOp.SUM)
    print(f"Tensor after AllReduce on rank {ptd.get_rank()}: {tensor}")
    
if __name__== '__main__':
    init_process()
    example_AllReduce()

'''
Tensor before AllReduce on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Tensor before AllReduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Tensor after AllReduce on rank 0:  tensor([3., 3., 3., 3., 3.], device='cuda:0')
Tensor after AllReduce on rank 1:  tensor([3., 3., 3., 3., 3.], device='cuda:1')
'''
# ==================================================

def example_reduce():
    rank   = ptd.get_rank()
    tensor = torch.tensor([rank + 1] * 5, dtype=torch.float32).cuda()
    print(f"Tensor before reduce on rank {rank}: {tensor}")
    ptd.reduce(tensor, dst=0, op=ptd.ReduceOp.SUM)
    print(f"Tensor after reduce on rank {rank}: {tensor}")
    
if __name__== '__main__':
    init_process()
    example_reduce()

'''
Tensor before reduce on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Tensor before reduce on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Tensor after reduce on rank 0:  tensor([3., 3., 3., 3., 3.], device='cuda:0')
Tensor after reduce on rank 1:  tensor([2., 2., 2., 2., 2.], device='cuda:1')
'''
# ==================================================

def example_gather():
    rank   = ptd.get_rank()
    tensor = torch.tensor([rank + 1] * 5, dtype=torch.float32).cuda()
    if rank == 0:
        gather_list = [
            torch.zeros(5, dtype=torch.float32).cuda()
            for _ in range(ptd.get_world_size())
            ]
    else:
        gather_list = None
    print(f"Tensor before gather on rank {rank}: {tensor}")
    ptd.gather(tensor, gather_list, dst=0)
    if rank == 0:
        print(f"List of tensors gathered by rank 0: {gather_list}")
    
if __name__== '__main__':
    init_process()
    example_gather()

'''
Tensor before gather on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Tensor before gather on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
List of tensors gathered by rank 0:
    [
    tensor([1., 1., 1., 1., 1.], device='cuda:0'), 
    tensor([2., 2., 2., 2., 2.], device='cuda:0')
    ]
'''
# ==================================================

def example_all_gather():
    rank   = ptd.get_rank()
    tensor = torch.tensor([rank + 1] * 5, dtype=torch.float32).cuda()
    gather_list = [
        torch.zeros(5, dtype=torch.float32).cuda()
        for _ in range(ptd.get_world_size())
        ]
    print(f"Tensor before AllGather on rank {rank}: {tensor}")
    ptd.all_gather(gather_list, tensor)
    print(f"List of tensors after AllGather on rank {rank}: {gather_list}")
    
if __name__== '__main__':
    grp = init_process()
    example_all_gather()
    ptd.destroy_process_group(grp)

'''
Tensor before AllGather on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
Tensor before AllGather on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
List of tensors after AllGather on rank 1: [tensor([1., 1., 1., 1., 1.], device='cuda:1'), tensor([2., 2., 2., 2., 2.], device='cuda:1')]
List of tensors after AllGather on rank 0: [tensor([1., 1., 1., 1., 1.], device='cuda:0'), tensor([2., 2., 2., 2., 2.], device='cuda:0')]
'''
# ==================================================

def example_scatter():
    rank = ptd.get_rank()
    if rank == 0:
        scatter_list = [
            torch.tensor([i + 1] * 5, dtype=torch.float32).cuda()
            for i in range(ptd.get_world_size())
            ]
        print(f"Rank 0 list of tensors to scatter: {scatter_list}")
    else:
        scatter_list = None
    tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f"Tensor before scatter on rank {rank}: {tensor}")
    ptd.scatter(tensor, scatter_list, src=0)
    print(f"Tensor after scatter on rank {rank}: {tensor}")
    
if __name__== '__main__':
    grp = init_process()
    example_scatter()
    ptd.destroy_process_group(grp)

'''
Rank 0 list of tensors to scatter: [
    tensor([1., 1., 1., 1., 1.], device='cuda:0'), 
    tensor([2., 2., 2., 2., 2.], device='cuda:0')]

Tensor before scatter on rank 0: tensor([0., 0., 0., 0., 0.], device='cuda:0')
Tensor before scatter on rank 1: tensor([0., 0., 0., 0., 0.], device='cuda:1')

Tensor after scatter on rank 0: tensor([1., 1., 1., 1., 1.], device='cuda:0')
Tensor after scatter on rank 1: tensor([2., 2., 2., 2., 2.], device='cuda:1')
'''
# ==================================================

def example_reduce_scatter():
    rank         = ptd.get_rank()
    world_size   = ptd.get_world_size()
    input_tensor = [
        torch.tensor([(rank + 1) * i for i in range(1, 3)], dtype=torch.float32).cuda()**(j+1) 
        for j in range(world_size)
        ]
    output_tensor = torch.zeros(2, dtype=torch.float32).cuda()
    print(f"List of tensors before ReduceScatter on rank {rank}: {input_tensor}")
    ptd.reduce_scatter(output_tensor, input_tensor, op=ptd.ReduceOp.SUM)
    print(f"Output tensor after ReduceScatter on rank {rank}: {output_tensor}")    
    
if __name__== '__main__':
    grp = init_process()
    example_reduce_scatter()
    ptd.destroy_process_group(grp)

'''
List of tensors before ReduceScatter on rank 1: [
    tensor([ 2.,  4.], device='cuda:1'), 
    tensor([ 4., 16.], device='cuda:1')]
List of tensors before ReduceScatter on rank 0: [
    tensor([1., 2.], device='cuda:0'), 
    tensor([1., 4.], device='cuda:0')]
Output tensor after ReduceScatter on rank 0: tensor([3., 6.], device='cuda:0')
Output tensor after ReduceScatter on rank 1: tensor([ 5., 20.], device='cuda:1')
'''
# ==================================================

def example_all2all():
    rank         = ptd.get_rank()
    world_size   = ptd.get_world_size()
    input_tensor = torch.arange(world_size) + rank * world_size
    input_tensor = input_tensor.cuda()
    print(f"Input tensor before All2All on rank {rank}: {input_tensor}")
    output_tensor = torch.empty([world_size], dtype=torch.int64).cuda()
    ptd.all_to_all_single(output_tensor, input_tensor)
    print(f"Output tensor after All2All on rank {rank}: {output_tensor}")

if __name__== '__main__':
    grp = init_process()
    example_all2all()
    ptd.destroy_process_group(grp)

'''
# considering three nodes
Input tensor before All2All on rank 0: tensor([0, 1, 2], device='cuda:0')
Input tensor before All2All on rank 1: tensor([3, 4, 5], device='cuda:1')
Input tensor before All2All on rank 2: tensor([6, 7, 8], device='cuda:2')

Output tensor after All2All on rank 0:  tensor([0, 3, 6], device='cuda:0')
Output tensor after All2All on rank 1:  tensor([1, 4, 7], device='cuda:1')
Output tensor after All2All on rank 1:  tensor([2, 5, 8], device='cuda:2')
'''
# ==================================================
