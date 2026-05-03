import torch_fidelity

'''
--------------------------------------------------------------------------
GIVEN A FOLDER WITH REAL IMAGES AND ANOTHER FOLDER WITH GENERATED IMAGES,
CALCULATES THE INCEPTION SCORE, THE FRECHET INCEPTION DISTANCE, AND THE
KERNEL INCEPTION DISTANCE.

TODO:
* Modify arguments 'input1 and 'input2' of 'torch_fidelity.calculate_metrics'.
--------------------------------------------------------------------------

isc_splits      (int): Number of splits in ISC. Default: `10`.
kid_subsets     (int): Number of subsets in KID. Default: `100`.
kid_subset_size (int): Subset size in KID. Default: `1000`.
batch_size      (int): Batch size used to process images. Default: `64`.

Returns a Dictionary of metrics with a subset of the following keys:
{
    'inception_score_mean':           1.964907712620493,
    'inception_score_std':            0.2501736823201583,
    'frechet_inception_distance':     0.5936997554067069,
    'kernel_inception_distance_mean': 0.0006927591754545004,
    'kernel_inception_distance_std':  6.297808508363855e-08
}
'''

metrics_dict = None

try:
    metrics_dict = torch_fidelity.calculate_metrics(
        input1='OUR_WORK_DIR_HERE/results/CFG_diffusion_04/generated_images',
        input2='OUR_WORK_DIR_HERE/results/cifar10test_5600_real_images',
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        kid_subset_size=32,
        batch_size=32,
        verbose=False,
        feature_extractor_internal_dtype='float64',
        feature_extractor="inception-v3-compat",
        samples_resize_and_crop=64,
        feature_layer_fid='192',
    )
except AssertionError as e:
    print(f'ERROR: {e}')

print(f"IS mean:  {metrics_dict['inception_score_mean']} sttdev: {metrics_dict['inception_score_std']}")
print(f"FID:      {metrics_dict['frechet_inception_distance']}")
print(f"KID mean: {metrics_dict['kernel_inception_distance_mean']} sttdev: {metrics_dict['kernel_inception_distance_std']}")
