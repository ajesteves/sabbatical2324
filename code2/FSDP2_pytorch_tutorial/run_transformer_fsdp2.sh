# /bin/bash
# bash run_transformer_fsdp2.sh {transformer_fsdp2.py} {num_gpus}
# in this case, the file to run is 'transformer_fsdp2.py'.
# num_gpus = num local gpus to use (must be at least 2).

echo "Launching ${1} with ${2} gpus"
torchrun --nnodes=1 --nproc_per_node=${2} ${1}

