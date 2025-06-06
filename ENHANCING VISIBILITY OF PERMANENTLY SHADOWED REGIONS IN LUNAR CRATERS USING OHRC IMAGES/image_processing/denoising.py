import cv2
import numpy as np
from skimage.restoration import denoise_wavelet, denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte

def apply_wavelet_denoising(image):
    """Apply Wavelet Denoising to reduce noise."""
    image_normalized = normalize_image(image)
    sigma_est = estimate_sigma(image_normalized, average_sigmas=True)
    denoised_image = denoise_wavelet(image_normalized, sigma=sigma_est, convert2ycbcr=False)
    return img_as_ubyte(denoised_image)

def apply_nlm_denoising(image):
    """Apply Non-Local Means Denoising."""
    image_normalized = normalize_image(image)
    sigma_est = np.mean(estimate_sigma(image_normalized))
    denoised_image_nlm = denoise_nl_means(image_normalized, h=1.15 * sigma_est, fast_mode=True)
    return img_as_ubyte(denoised_image_nlm)

def apply_bilateral_filter(image):
    """Apply Bilateral Filtering to smooth while preserving edges."""
    image_ubyte = img_as_ubyte(image)
    return cv2.bilateralFilter(image_ubyte, d=9, sigmaColor=75, sigmaSpace=75)