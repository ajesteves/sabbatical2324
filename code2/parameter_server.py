'''
To implement the parameter server, we start by importing the necessary modules 
and by defining a simple convolutional neural network that we will train on 
the MNIST dataset.
'''
import argparse
import os
from pydoc import Helper
import time
from   threading import Lock

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from   torch import optim
from   torch.distributed.optim import DistributedOptimizer
from   torchvision import datasets, transforms

'''
----------------------------------------------------------------------
Neural network to classify MNIST images
----------------------------------------------------------------------
'''
class Net(nn.Module):
   def __init__(self, num_gpus=0):
       super(Net, self).__init__()
       print(f"Using {num_gpus} GPUs for training")
       self.num_gpus = num_gpus
       device = torch.device(
           "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
       print(f"Place the first 2 convolutions on {str(device)}")
       # Place the convolution layers on the first GPU, or CPU if there is no GPU
       self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
       self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
       # Place the remaining layers on the 2nd GPU, if there is one
       if "cuda" in str(device) and num_gpus > 1:
           device = torch.device("cuda:1")

       print(f"Place the remaining layers on {str(device)}")
       self.dropout1 = nn.Dropout2d(0.25).to(device)
       self.dropout2 = nn.Dropout2d(0.5).to(device)
       self.fc1      = nn.Linear(9216, 128).to(device)
       self.fc2      = nn.Linear(128, 10).to(device)

   def forward(self, x):
       x = self.conv1(x)
       x = F.relu(x)
       x = self.conv2(x)
       x = F.max_pool2d(x, 2)

       x = self.dropout1(x)
       x = torch.flatten(x, 1)
       # Move tensor 'x' to the next device if necessary
       next_device = next(self.fc1.parameters()).device
       x = x.to(next_device)

       x      = self.fc1(x)
       x      = F.relu(x)
       x      = self.dropout2(x)
       x      = self.fc2(x)
       output = F.log_softmax(x, dim=1)
       return output

"""
----------------------------------------------------------------------
Helper functions that will be useful for the rest of our parameter server. This code uses 
rpc_sync and RRef for defining a function that invokes a given method on an object living 
on a remote node. The handle to the remote object is given by the 'rref' argument 
of the functions, and we run it on its owning node, 'rref.owner()'. On the caller node, 
we run this command synchronously through the use of 'rpc_sync', meaning that we block 
the caller until a response is received.

On the local node, call a method with first 'arg' as the value held by the RRef.
Other 'arg's are passed in as arguments to the function called. This is useful 
for calling instance methods. 'method' can be any matching function, including 
class methods.
"""
def call_method(method, rref, *args, **kwargs):
   return method(rref.local_value(), *args, **kwargs)

"""
Given an RRef, returns the result of calling the provided method on the value
held by the RRef. This call is done on the remote node that owns the RRef and 
passes along the given argument.
Example: If the value held by the RRef is of type 'Foo', then
'remote_method(Foo.bar, rref, arg1, arg2)' is equivalent to calling
'<foo_instance>.bar(arg1, arg2)' on the remote node and getting the result back.
"""
def remote_method(method, rref, *args, **kwargs):
   args = [method, rref] + list(args)
   return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

"""
----------------------------------------------------------------------
Parameter server class. 
It inherits 'nn.Module'.
"""
class ParameterServer(nn.Module):
    """
    Instantiates the neural network defined above.
    Registers the identification of device that will be used for placing the network.
    """
    def __init__(self, num_gpus=0):
        super().__init__()
        model             = Net(num_gpus=num_gpus)
        self.model        = model
        self.input_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu"
        )

    """
    Regardless of the device where the model is trained, we move the output to CPU, 
    as the Distributed RPC framework only supports sending CPU tensors over RPC. 
    We have intentionally disabled sending CUDA tensors over RPC to avoid having 
    different devices on the caller/callee.
    """
    def forward(self, inp):
        inp = inp.to(self.input_device)
        out = self.model(inp)
        # The output tensor is moved to CPU
        out = out.to("cpu")
        return out

    """
    Next, we create two methods that are useful for training and verification 
    purposes. The 'get_dist_gradients' method gets a Distributed Autograd 
    context identifier as argument and calls ''dist_autograd.get_gradients' 
    in order to retrieve the gradients computed by the distributed Autograd. 
    It also iterates through the resulting dictionary with gradients and 
    converts each tensor to a CPU tensor, as PyTorch might not support sending 
    tensors over RPC yet. The 'get_param_rrefs' method iterates through our 
    model weights and wrap them with local RRefs. This method will be called 
    over RPC by the worker nodes and returns a list of the weights to be 
    optimized. This is required as input to the distributed optimizer, which 
    requires the weights to optimize as a list of RRefs.

    Use distributed Autograd to retrieve gradients accumulated for this model.
    Primarily used for verification.
    """
    def get_dist_gradients(self, contextID):
        grads = dist_autograd.get_gradients(contextID)
        # This output is forwarded over RPC, and actual PyTorch might not yet accept GPU tensors.
        # Thus, tensors are moved from GPU memory to CPU.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    # Wrap the local weights with RRefs. This is necessary for building the
    # DistributedOptimizer that will optimize the weights remotely.
    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs


"""
----------------------------------------------------------------------
Create two methods that helps us initialize the parameter server. 
Note that there will only be one instance of the parameter server 
across all processes, all workers will interact with the same parameter 
server and update the same weights that it stores. As seen in 
'run_parameter_server', the server itself does not take any independent 
actions. It waits for requests from workers, which we will define below, 
and responds to them by running the requested function.
"""
# The global parameter server instance.
param_server = None

# A lock to ensure that we only have one parameter server.
global_lock = Lock()

def get_parameter_server(num_gpus=0):
    """
    Returns a singleton parameter server to all trainer processes.
    """
    global param_server
    # Ensure that we get only one handle to the parameter server.
    with global_lock:
        if not param_server:
            # Instantiate the ParameterServer
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server

def run_parameter_server(rank, world_size):
    """
    The parameter server just acts as a host for our model and responds to
    the requests received from the workers.
    'rpc.shutdown()' waits for all workers to complete, which in this case
    means that the parameter server will wait for all workers to complete,
    and then exits.
    'rpc.shutdown()' will not immediately shut down the parameter server. Instead, 
    it waits for all workers to also call 'rpc.shutdown()'. This gives us the 
    guarantee that the parameter server will not go offline before all workers 
    have completed their training task.
    """
    print("Parameter server master process initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC initialized! Running the parameter server...")
    rpc.shutdown()
    print("RPC shutdown on the parameter server.")


"""
----------------------------------------------------------------------
The worker class.
'nn.Module' corresponds to the network that will be trained by this worker.
"""
class WorkerNet(nn.Module):
    """
    The constructor method uses 'rpc.remote' to obtain a remote reference (an RRef) to
    the parameter server. We are not copying the parameter server to the local process, 
    instead, we can think of 'self.param_server_rref' as a distributed shared pointer to
    the parameter server that lives on a separate process.
    """
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote(
            "parameter_server", 
            get_parameter_server, 
            args = (num_gpus,)
        )

    """
    This method is necessary because the 'DistributedOptimizer' requires as input 
    a list of RRefs corresponding to the remote weights to be optimized, so here we 
    obtain these RRefs. Since the only remote process that a given 'WorkerNet' 
    interacts with is the 'ParameterServer', we simply invoke a 'remote_method' 
    on the 'ParameterServer'. This method uses the 'get_param_rrefs' method of
    the 'ParameterServer' class. The method returns a list of RRefs to the weights 
    that need to be optimized. In this case, our 'WorkerNet' does not define its 
    own weights; if it did, we would need to wrap each weight with an RRef as well 
    and include it into our input to the Distributed Optimizer.
    """
    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref
        )
        return remote_params

    """
    The 'forward' method of the worker class uses a synchronous RPC call to run 
    the forward pass of model training, which corresponds to the forward} method
    of the 'ParameterServer' class. In this RPC call, we pass the 
    'self.param_server_rref' as argument, which is a remote handle to the 
    'ParameterServer'. This call sends an RPC to the node on which our 
    'ParameterServer' is running, invokes the forward} pass, and returns the 
    Tensor corresponding to the model's output.
    """
    def forward(self, x):
        model_output = remote_method(
            ParameterServer.forward, 
            self.param_server_rref, 
            x
        )
        return model_output


"""
-----------------------------------------------------------------------
The next step is to implement the loop for training our model, where we pass 
some input data through the network and compute the loss. The training loop 
is similar to the sequential version, with some modifications due to the 
nature of our network being distributed across machines.

As mentioned above, we must apply into the optimizer all of the global weights, 
global across all nodes participating in distributed training, that need to be 
optimized. Additionally, we also provide as input the local optimizer to be used, 
which in this case is SGD. We can configure the underlying optimizer algorithm 
in the same way as we do with a local optimizer. All arguments of optim.SGD} 
will be provided by the DistributedOptimizer}. As an example, we provide the 
'DistributedOptimizer' with a custom learning rate of $0.03$ that will be used 
as the learning rate by all local SGD optimizers.
"""
def run_training_loop(rank, num_gpus, train_loader, test_loader):
    # Runs the typical neural network forward + backward + optimizer steps,
    # Instantiate the worker
    net = WorkerNet(num_gpus=num_gpus)
    print(f'Worker {rank} instantiated the neural network')

    # Instantiate the distributed optimizer
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)
    print(f'Worker {rank} instantiated the distributed optimizer')

    """
    The training loop iterates over the provided training DataLoader. Before the 
    forward-backward-optimizer steps, we first create a distributed Autograd 
    context to run them. This is necessary to record RPCs invoked in the model's 
    forward pass, so that an appropriate graph can be constructed, which includes 
    all participating distributed workers in the backward pass. The distributed 
    Autograd context returns a contextID} that serves as an identifier for 
    accumulating and optimizing gradients corresponding to a particular iteration.

    As opposed to calling 'loss.backward()', which would run the backward pass on
    a local worker, we call 'dist_autograd.backward()' and pass as input the 
    'contextID' as well as the 'loss'. In addition, we pass the 'contextID' as 
    input of the optimizer call, which is required to be able to look up the
    corresponding gradients computed by this particular backward pass across all nodes.
    """
    for i, (data, target) in enumerate(train_loader):
        with dist_autograd.context() as contextID:
            model_output = net(data)
            print(f"Rank {rank} :: forward pass done")
            target = target.to(model_output.device)
            loss   = F.nll_loss(model_output, target)
            print(f"Rank {rank} :: calculate loss done")
            dist_autograd.backward(contextID, [loss])
            print(f"Rank {rank} :: backward pass done")
            # Ensure that the distributed Autograd ran successfully and
            # the gradients were returned.
            assert remote_method(
                ParameterServer.get_dist_gradients,
                net.param_server_rref,
                contextID,
            ) != {}
            print(f"Rank {rank} :: assert passed")
            opt.step(contextID)
            #if i % 5 == 0:
            print(f"Rank {rank} :: training batch {i} :: loss {loss.item()}")

    print("Training complete!")
    print("Getting accuracy....")
    get_accuracy(test_loader, net)

"""
-----------------------------------------------------------------------
The function 'get_accuracy' computes the test accuracy of the model after 
it has been trained, much like in sequential training.
"""
def get_accuracy(test_loader, model):
    model.eval() # Put the model in evaluation mode
    correct_sum = 0
    # Use a GPU in evaluation if available
    device = torch.device("cuda:0" if model.num_gpus > 0
        and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out          = model(data, -1)
            pred         = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct      = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f'Accuracy {correct_sum / len(test_loader.dataset)}')

"""
-----------------------------------------------------------------------
Just as we defined a main loop in 'run_parameter_server' for the parameter 
server, which is responsible for initializing RPC, we define a similar 
loop for the workers. The difference is that the workers must run the 
training loop defined above. Analogously to 'run_parameter_server', 
by default 'rpc.shutdown()' waits for all processes, both workers and 
parameter server, to call 'rpc.shutdown()' before each worker exits. 
This ensures that the nodes are terminated gracefully and no node goes 
offline while another is expecting it to be online.

Main loop of the worker.
"""
def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
    print(f'Worker rank {rank} initializing RPC')
    rpc.init_rpc(
        name       = f'trainer_{rank}',
        rank       = rank,
        world_size = world_size
    )

    print(f'Worker {rank} finished initializing RPC')

    run_training_loop(rank, num_gpus, train_loader, test_loader)
    rpc.shutdown()

"""
-----------------------------------------------------------------------
We completed the implementation of worker and parameter server, and all that is left 
is the code to launch the workers and the parameter server. First, we must provide
the arguments necessary to instantiate the parameter server and workers. 'world_size' 
corresponds to the total number of nodes that will participate in training, and 
accounts all workers and the parameter server. It is also necessary to provide 
a unique 'rank' for each individual process, from 0 (where we will run the single 
parameter server) to 'world_size - 1'. 'master_addr' and 'master_port' are arguments 
that can be used to identify where the rank 0 process is running, and will be used 
by individual nodes to discover each other. To test the implementation locally, 
we can simply use 'localhost' as 'master_addr' and a common 'master_port' for all 
instances spawned. Our implementation only supports between 0 and 2 GPUs, but can 
be extended to allow more GPUs.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all worker nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Use 0 for master.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, Currently we support between 0
        and 2 GPUs. Note that this argument will be passed to the parameter server.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address:port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that the master listens to, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    assert args.num_gpus <= 3, f"Only 0-2 GPUs are currently supported, got {args.num_gpus}."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    """
    Finally, we create a process corresponding to either a parameter server or a worker, 
    depending on the command line arguments. It is created a 'ParameterServer' if 
    we provide as rank the value $0$, and it is created a WorkerNet} otherwise. 
    We are using 'torch.multiprocessing' to launch a subprocess corresponding to the 
    function that we want to execute, and waiting on this process's completion from 
    the main thread with 'p.join()'. If we initialize a worker, we also provide as 
    arguments the training and test 'DataLoader's for the MNIST dataset.
    """
    processes = []
    world_size = args.world_size

    if args.rank == 0:
        # start a parameter server on this process
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    else:
        # Get the training data
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data',
                train     = True,
                download  = True,
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]
                )
            ),
            batch_size = 32, 
            shuffle    = True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data',
                train     = False,
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]
                )
            ),
            batch_size = 32, 
            shuffle    = True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data',
                train     = False,
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]
                )
            ),
            batch_size = 32,
            shuffle    = True,
        )
        # Start a training worker on this process
        p = mp.Process(
            target = run_worker,
            args   = (
                args.rank,
                world_size, 
                args.num_gpus,
                train_loader,
                test_loader
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

"""
-----------------------------------------------------------------------
To run the code locally, assuming that it is saved in the parameter_server.py 
file, we use the following command for the server and each worker that we want to 
spawn, in separate operating system terminals:

python parameter_server.py --world_size=WORLD_SIZE --rank=RANK

For example, for a master node with a world size of 2, the command would be:

python parameter_server.py --world_size=2 --rank=0

The worker can then be launched with the following command, issued on a separate terminal:

python parameter_server.py --world_size=2 --rank=1 

This creates a training session with one server and a single worker.

We can also include in the previous commands the arguments --master_addr=<ADDRESS> 
and --master_port=<PORT> to indicate the address and port that the master worker 
is listening to, for example, to evaluate a scenario where workers and master node 
run on different machines.
"""
