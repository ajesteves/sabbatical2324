import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

#--------------------------------------------------------------------------
# GIVEN A NPY FILE WITH REAL IMAGES AND ANOTHER NPY FILE WITH GENERATED IMAGES,
# CALCULATES THE FRECHET INCEPTION DISTANCE.
# --------------------------------------------------------------------------

def load_npy_as_tensor(file_images):
    # Load the numpy array
    np_array = np.load(file_images)

    # Convert to PyTorch tensor with dtype uint8
    tensor = torch.tensor(np_array[0:2000,:,:,:], dtype=torch.uint8) # REPOR --> 5600
    tensor = torch.permute(tensor, (0, 3, 1, 2))

    tensor = torch.nn.functional.interpolate(tensor, size=(299,299))

    return tensor


file_real_images      = 'OUR_WORK_DIR_HERE/results/cifar10test_5600_real_images.npy'  
file_generated_images = 'OUR_WORK_DIR_HERE/results/CFG_diffusion_04/samples_5600_epoch_100_w_1.0.npy'

real_images = load_npy_as_tensor(file_real_images)
gen_images  = load_npy_as_tensor(file_generated_images)

print(f'Real images shape: {real_images.shape} type: {real_images.dtype}')
print(f'Generated images shape: {gen_images.shape} type: {gen_images.dtype}')

fid = FrechetInceptionDistance(feature=2048, compute_on_cpu=True) # value from {64, 192, 768, 2048}

fid.update(real_images, real=True)
fid.update(gen_images,  real=False)
fid_value = fid.compute()

print(f'FID: {fid_value}')
