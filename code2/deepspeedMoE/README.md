This code demonstrates how to apply DeepSpeed-MoE to a simple CIFAR-10 model.

`cifar10_tutorial.py`, `cifar10_tutorial.ipynb`

    Baseline CIFAR-10 model.

`cifar10_deepspeed.py`
    DeepSpeed-MoE added to the base CIFAR-10 model.

`run_ds.sh`
    Script for running DeepSpeed base model.

`run_ds_moe.sh`
    Script for running DeepSpeed model with Mixture of Experts (MoE) integration.

`run_ds_prmoe.sh`
    Script for running DeepSpeed model with Pyramid Residual MoE (PR-MoE) integration.

* To run baseline CIFAR-10 model - `python cifar10_tutorial.py`
* To run DeepSpeed CIFAR-10 model - `bash run_ds.sh`
* To run DeepSpeed CIFAR-10 model with Mixture of Experts (MoE) - `bash run_ds_moe.sh`
* To run DeepSpeed CIFAR-10 model with Pyramid Residual MoE (PR-MoE) - `bash run_ds_prmoe.sh`
* To run with different data type (default=`fp16`) and zero stages (default=`0`) - `bash run_ds.sh --dtype={fp16|bf16} --stage={0|1|2|3}`
