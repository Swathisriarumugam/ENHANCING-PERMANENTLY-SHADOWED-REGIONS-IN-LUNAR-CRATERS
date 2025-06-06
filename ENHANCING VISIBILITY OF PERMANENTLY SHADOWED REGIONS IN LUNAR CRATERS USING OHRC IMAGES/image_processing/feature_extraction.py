import cv2
import numpy as np
from skimage.feature import local_binary_pattern, canny
from skimage.filters import gabor
from skimage import img_as_ubyte

def apply_canny_edge_detection(image, sigma=1.0):
    """Apply Canny Edge Detection to detect edges."""
    edges = canny(image, sigma=sigma)
    return img_as_ubyte(edges)

def apply_gabor_filter(image, frequency=0.6):
    """Apply Gabor Filter to extract texture features."""
    filt_real, _ = gabor(image, frequency=frequency)
    return img_as_ubyte(filt_real)

def apply_lbp(image, radius=1, n_points=8):
    """Apply Local Binary Pattern (LBP) to extract texture features."""
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    lbp_normalized = (lbp - np.min(lbp)) / (np.max(lbp) - np.min(lbp))
    return img_as_ubyte(lbp_normalized)