"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import yaml
import pickle
from   contextlib import nullcontext
import numpy      as     np
import torch
import torch._dynamo
from   torch.nn.parallel import DistributedDataParallel as DDP
from   torch.distributed import init_process_group, destroy_process_group

from   model import MOEconfig, Transformer

torch._dynamo.config.suppress_errors = True

# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'

# -----------------------------------------------------------------------------
# Read configuration from yaml file

CONFIG_FILE = 'config/train_moe_v1.yaml'

with open(CONFIG_FILE, 'r') as file:
    try:
        config = yaml.safe_load(file)
        config["gradient_accumulation_steps"] *= config["n_gpu"]
        config["wandb_run_name"] = config["wandb_run_name"] + time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
        config["dtype"] = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    except yaml.YAMLError as exc:
        print(exc)

if int(os.environ['RANK']) == 0:
    print('::::::::::::::::: configuration ::::::::::::::::::')
    for key, value in config.items():
        print(f'\t{key}: {value}')

# -----------------------------------------------------------------------------
# Various initializations, derived attributes, input/output setup

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=config["backend"])
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device         = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # Rank 0 process will do logging, checkpointing etc.
    seed_offset    = ddp_rank      # Each process gets a different seed.
    # 'world_size' is the number of processes that we will use fortraining simultaneously,
    # so we can scale down the desired gradient accumulation iterations per process 
    # proportionally
    assert config["gradient_accumulation_steps"] % ddp_world_size == 0
    config["gradient_accumulation_steps"] //= ddp_world_size
else:
    # If not using DDP, we are running on a single gpu, and one process
    master_process = True
    seed_offset    = 0
    ddp_world_size = 1

tokens_per_iter = config["gradient_accumulation_steps"] * ddp_world_size * config["batch_size"] * config["block_size"]
print(f"[INFO] Tokens per iteration will be {tokens_per_iter:,}")

if master_process:
    os.makedirs(config["out_dir"], exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32       = True # allow tf32 on cudnn

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# float16 data type will automatically use a GradScaler
ptdtype = {
    'float32':  torch.float32, 
    'bfloat16': torch.bfloat16, 
    'float16':  torch.float16
    }[config["dtype"]]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type = device_type, 
    dtype       = ptdtype
)

# -----------------------------------------------------------------------------
# poor man's data loader

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as indicated in
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(
            os.path.join(config["data_path"], 'train.bin'), 
            dtype = np.uint16, 
            mode  = 'r',
        )
    else:
        data = np.memmap(
            os.path.join(config["data_path"], 'val.bin'), 
            dtype = np.uint16, 
            mode  = 'r',
        )

    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],)) 
    x  = torch.stack([torch.from_numpy((data[i:i+config["block_size"]]).astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy((data[i+1:i+1+config["block_size"]]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    # 'x' shape = 'y' shape = [BATCH SIZE, BLOCK SIZE] = [12, 1024]
    return x, y

# Initialize these parameters here.
# They can overriden if init_from='resume', by read them from a checkpoint
iter_num      = 0
best_val_loss = 1e9

# -----------------------------------------------------------------------------
# Attempt to derive vocabulary size from the dataset
meta_path       = os.path.join(config["data_path"], 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"[INFO] Found vocabulary size = {meta_vocab_size} (inside {meta_path})")

# -----------------------------------------------------------------------------
# Model initialization

# Define the model configuration parameters
model_args = dict(
    n_layer               = config["n_layer"], 
    n_head                = config["n_head"], 
    n_embd                = config["n_embd"], 
    block_size            = config["block_size"],
    bias                  = config["bias"], 
    vocab_size            = None, 
    dropout               = config["dropout"], 
    n_exp                 = config["n_exp"], 
    top_k                 = config["top_k"],
    use_aux_loss          = config["use_aux_loss"], 
    use_router_z_loss     = config["use_router_z_loss"],
    use_noisy_top_k       = config["use_noisy_top_k"], 
    aux_loss_weight       = config["aux_loss_weight"],
    router_z_loss_weight  = config["router_z_loss_weight"], 
    train_capacity        = config["train_capacity"],
    eval_capacity         = config["eval_capacity"], 
    min_capacity          = config["min_capacity"], 
    stride                = config["stride"],
    use_switch_tfm_init   = config["use_switch_tfm_init"], 
    switch_tfm_init_scale = config["switch_tfm_init_scale"],
    router_use_full_prec  = config["router_use_full_prec"],
)

print('\n\n')
print(model_args)
print('\n\n')

# -----------------------------------------------------------------------------
if config["init_from"] == 'scratch':
    # Instantiate a new model from scratch
    print("[INFO] Initializing a new model from scratch")
    
    # Determine the vocabulary size we will use for from-scratch training
    if meta_vocab_size is None:
        print("[INFO] Defaulting to vocabulary size of GPT-2 equal to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    
    # Create the MoE configuration and the model
    moeconf = MOEconfig(**model_args)
    model   = Transformer(moeconf)

elif config["init_from"] == 'resume':
    print(f"[INFO] Resuming training from {config['out_dir']}")
    # Resume training from a checkpoint
    ckpt_path             = os.path.join(config["out_dir"], 'ckpt.pt')
    checkpoint            = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # Force these configuration attributes to be equal, otherwise we can not 
    # resume training; the rest of the attributes (e.g. dropout) can stay 
    # as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    # Create the MoE configuration and the model
    moeconf   = MOEconfig(**model_args)
    model     = Transformer(moeconf)

    # Load checkpoint from file
    state_dict = checkpoint['model']
    
    # Fix the keys of the state dictionary
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # Load weights from checkpoint into the model
    model.load_state_dict(state_dict)

    # Set the iteration number and best validation loss that we read 
    # from the checkpoint, so that the training loop can resume from there
    iter_num      = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif config["init_from"].startswith('gpt2'):
    print(f"[INFO] Initializing the model from OpenAI GPT-2 weights: {config['init_from']}")
    # Initialize our model with OpenAI GPT-2 weights
    override_args = dict(dropout=config["dropout"])
    model         = Transformer.from_pretrained(config["init_from"], override_args)

    # Read the configuration parameters, from the model that we just created 
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# Crop down the model block size if desired
# so that the checkpoint will have the right value
if config["block_size"] < model.config.block_size:
    model.crop_block_size(config["block_size"])
    model_args['block_size'] = config["block_size"] 

model.to(device)

# Initialize a GradScaler. If enabled=False, GradScaler does nothing.
scaler = torch.amp.GradScaler('cuda', enabled=(config["dtype"] == 'float16'))

# Setup the optimizer
optimizer = model.configure_optimizers(
    config["weight_decay"], 
    config["learning_rate"], 
    (config["beta1"], config["beta2"]), 
    device_type
)
if config["init_from"] == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None # free up memory

# Compile the model
if config["compile"]:
    print("[INFO] compiling the model... (takes around 1 minute)")
    unoptimized_model = model
    model             = torch.compile(model) # requires PyTorch 2.0

# Wrap the model with a DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Learning rate decay scheduler: cosine with warmup
def get_lr(it):
    # 1) linear warmup during 'warmup_iters' steps
    if it < config["warmup_iters"]:
        return config["learning_rate"] * (it + 1) / (config["warmup_iters"] + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    # 3) in between, use cosine decay down to the minimum learning rate
    decay_ratio = (it - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 'coeff' ranges in 0..1
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

# -----------------------------------------------------------------------------
# WandB logging
#
# Use this block to start training from the beginning ................
if config["wandb_log"] and master_process:
    import wandb
    wandb.init(
        project = config["wandb_project"], 
        name    = config["wandb_run_name"], 
        config  = config
    )
#
# Use this block to resume a previous run ............................
#if config["wandb_log"] and master_process:
#    config["wandb_run_name"] = "moe-base-2026-06-16-20h23m26s"
#    import wandb
#    wandb.init(
#        project = config["wandb_project"], 
#        name    = config["wandb_run_name"], 
#        config  = config,
#        resume  = True,
#        id      = "ysxh05we"
#    )

# --------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------

X, Y           = get_batch('train') # fetch the very first batch
t0             = time.time()
local_iter_num = 0        # number of iterations in the lifetime of this process
raw_model      = model.module if ddp else model # unwrap DDP container if needed
running_mfu    = -1.0

while True:
    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config["decay_lr"] else config["learning_rate"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on training/validation sets and write checkpoint
    if iter_num % config["eval_interval"] == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if config["wandb_log"]:
            wandb.log({
                "iteration": iter_num,
                "train_loss": losses['train'],
                "val_loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or config["always_save_checkpoint"]:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model':         raw_model.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'model_args':    model_args,
                    'iter_num':      iter_num,
                    'best_val_loss': best_val_loss,
                    'config':        config,
                }
                print(f"[INFO] saving checkpoint to {config['out_dir']}")
                torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))

    if iter_num == 0 and config["eval_only"]:
        break

    # Forward pass and backward update, with optional gradient accumulation to
    # simulate a larger batch size and using the GradScaler if data type is float16.
    for micro_step in range(config["gradient_accumulation_steps"]):
        if ddp:
            # In DDP training we only need to sync gradients at the last micro step.
            # The official way to do this is with model.no_sync() context manager.
            # By looking at the source code of the context manager, it just toggles 
            # the following variable:
            model.require_backward_grad_sync = (micro_step == config["gradient_accumulation_steps"] - 1)
        with ctx:
            logits, loss = model(X, Y)
            # Scale the loss to account for gradient accumulation
            loss         = loss / config["gradient_accumulation_steps"]

        # Prefetch next batch asynchronously, while the model is doing the forward pass on GPU
        X, Y = get_batch('train')

        # Backward pass, with gradient scaling if training in float16
        scaler.scale(loss).backward()

    # Clip the gradients
    if config["grad_clip"] != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

    # Step the optimizer and scaler if training in float16
    scaler.step(optimizer)
    scaler.update()

    # Flush the gradients as soon as we can, no need for this in memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config["log_interval"] == 0 and master_process:
        # Get the loss as float.
        # Note: this is a CPU-GPU synchronization point.
        # Scale up to undo the division above, approximating the true total loss 
        # (exact would have been a sum)
        lossf = loss.item() * config["gradient_accumulation_steps"]
        if local_iter_num >= 5: # let the training loop to settle a bit
            mfu         = raw_model.estimate_mfu(config["batch_size"] * config["gradient_accumulation_steps"], dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"Iteration {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num       += 1
    local_iter_num += 1

    # Training termination condition
    if iter_num > config["max_iters"]:
        break

# -----------------------------------------------------------------------------
if ddp:
    destroy_process_group()
