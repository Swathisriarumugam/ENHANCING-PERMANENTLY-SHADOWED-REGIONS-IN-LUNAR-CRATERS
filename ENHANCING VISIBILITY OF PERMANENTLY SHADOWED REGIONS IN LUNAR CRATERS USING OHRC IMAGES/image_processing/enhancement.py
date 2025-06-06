import cv2
import numpy as np
from skimage import img_as_ubyte

def normalize_image(image):
    """Normalize image array to the range [0, 1]."""
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def apply_clahe(image):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    if image.dtype != np.uint8:
        image = img_as_ubyte(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def multi_scale_retinex(image, sigma_list=None):
    """Apply Multi-Scale Retinex enhancement."""
    if sigma_list is None:
        sigma_list = [15, 80, 250]
    
    retinex = np.zeros_like(image, dtype=np.float64)
    for sigma in sigma_list:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma).astype(np.float64)
        retinex += np.log10(image + 1.0) - np.log10(blurred + 1.0)
    
    retinex /= len(sigma_list)
    return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def gamma_correction(image, gamma=1.2):
    """Apply Gamma Correction to the image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)