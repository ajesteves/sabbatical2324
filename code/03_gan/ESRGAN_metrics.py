# ..............................................................................
# Calculate evaluation metrics with a trained super-resolution ESRGAN model.
# The metrics are the peak signal-to-noise ratio (PSNR) and the structural
# similarity (SSIM).
#
# Antonio Esteves @ UMinho, Aug 2024
# ..............................................................................

import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse

def rgb2ycbcr_pt(img, y_only=False):
    """
    Convert RGB images to YCbCr images.

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor):   Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
        y_only (bool):  Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight  = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight  = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias    = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img     = out_img / 255.
    return out_img


def _convert_input_type_range(img):
    """
    Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in color space
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img      = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """
    Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in color space conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """
    Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img      = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def reorder_image(img, input_order='HWC'):
    """
    Reorder images to 'H,W,C'.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """
    Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def calculate_psnr(img, img2, crop_border, test_y_channel=False, **kwargs):
    """
    Calculate the Peak Signal-to-Noise Ratio ( PSNR).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor):  Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img  = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img  = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img  = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse  = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


def _ssim_pth(img, img2):
    """
    Calculate the structural similarity (SSIM).
    This is the function that makes the real calculations.

    Args:
        img (Tensor):  Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result.
    """
    c1        = (0.01 * 255)**2
    c2        = (0.03 * 255)**2

    kernel    = cv2.getGaussianKernel(11, 1.5)
    window    = np.outer(kernel, kernel.transpose())
    window    = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1       = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2       = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq    = mu1.pow(2)
    mu2_sq    = mu2.pow(2)
    mu1_mu2   = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12   = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map    = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map  = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


def calculate_ssim(img, img2, crop_border, test_y_channel=False, **kwargs):
    """
    Calculate the structural similarity (SSIM).

    Paper: Image quality assessment: From error visibility to structural similarity.

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (Tensor):  Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img  = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img  = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img  = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """
    Convert a numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def eval_metrics(img_path, img_path2, crop_border, test_y_channel=False):
    '''
    Calculate the Peak Signal-to-Noise Ratio and 
    the structural similarity between two images.

    Args:
        img_path (string):     Path to the first image.
        img_path2 (string):    Path to the second image.
        crop_border (int):     Cropped pixels in each edge of the images. 
                               These pixels are not involved in the calculations.
        test_y_channel (bool): Make calculations on Y channel of YCbCr.
    Returns:
        tuple(float,float): peak signal-to-noise ratio and the structural similarity.
    '''
    img  = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)  # use 'cv2.IMREAD_COLOR' if the image has 4 color channels
    img2 = cv2.imread(img_path2, cv2.IMREAD_UNCHANGED)  # the image has 3 color channels

    img  = img2tensor(img / 255.,  bgr2rgb=True, float32=True).unsqueeze_(0)
    img2 = img2tensor(img2 / 255., bgr2rgb=True, float32=True).unsqueeze_(0)

    img      = img.cuda()
    img2     = img2.cuda()
    psnr_pth = calculate_psnr(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
    ssim_pth = calculate_ssim(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
    print(f'Image {img_path} \t\tPSNR: {psnr_pth[0]:.6f} dB, \tSSIM: {ssim_pth[0]:.6f}')

    return psnr_pth[0], ssim_pth[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--sr_dir', type=str, help='SR images path')
    parser.add_argument('--hr_dir', type=str, help='HR images path')
    args = parser.parse_args()

    if args.sr_dir == None or args.hr_dir == None:
        print('Necessary argument(s) not provided')
    else:
        # Examples of SR and HR image paths:
        #
        #sr_dir = './results/ESRGAN_Div2k_Flickr2k_06_stage2/test_div2k_300_interpolated_generator_alpha_0_90/sr'
        #sr_dir = './results/ESRGAN_Div2k_Flickr2k_06_stage2/test_div2k_300_gan_generator/sr'
        #hr_dir = 'OUR_DATASETS_DIR/div2k_300/hr'

        #sr_dir = './results/ESRGAN_Div2k_Flickr2k_02_stage2/test_set14_interpolated_generator_alpha_0_00/sr'
        #sr_dir = './results/ESRGAN_Div2k_Flickr2k_02_stage2/test_set14_gan_generator/sr'
        #hr_dir = 'OUR_DATASETS_DIR/Set14/GTmod12'

        psnr1 = 0.0
        ssim1 = 0.0
        psnr2 = 0.0
        ssim2 = 0.0
        count = 0

        for image_name in os.listdir(args.hr_dir):
            hr_img_path = args.hr_dir + '/' + image_name
            sr_img_path = args.sr_dir + '/' + image_name

            p1, s1 = eval_metrics(
                hr_img_path,
                sr_img_path,
                crop_border=4,
                test_y_channel=False
            )

            p2, s2 = eval_metrics(
                hr_img_path,
                sr_img_path,
                crop_border=4,
                test_y_channel=True
            )
            psnr1 += p1
            ssim1 += s1
            psnr2 += p2
            ssim2 += s2
            count += 1

        psnr1 /= count
        ssim1 /= count
        psnr2 /= count
        ssim2 /= count

        print(f'[Calculations on R,G,B channels]     Average PSNR: {psnr1:.6f} dB, \tAverage SSIM: {ssim1:.6f}')
        print(f'[Calculations on Y channel of YCbCr] Average PSNR: {psnr2:.6f} dB, \tAverage SSIM: {ssim2:.6f}')
