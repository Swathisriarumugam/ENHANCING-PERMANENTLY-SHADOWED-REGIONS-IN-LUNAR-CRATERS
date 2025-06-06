import os
import cv2
from image_processing.enhancement import *
from image_processing.denoising import *
from image_processing.feature_extraction import *
from image_processing.visualization import *
from image_processing.utils import create_directory

def process_image(image_path, processed_images_path):
    """Process an image with various techniques."""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Enhancement pipeline
        image_clahe = apply_clahe(image)
        image_msr = multi_scale_retinex(image_clahe)
        image_gamma = gamma_correction(image_msr, gamma=1.2)
        
        # Denoising pipeline
        image_denoised_wavelet = apply_wavelet_denoising(image_gamma)
        image_denoised_nlm = apply_nlm_denoising(image_denoised_wavelet)
        image_denoised_bilateral = apply_bilateral_filter(image_denoised_nlm)
        
        # Feature extraction
        edges = apply_canny_edge_detection(image_denoised_bilateral)
        gabor_texture = apply_gabor_filter(image_denoised_bilateral)
        lbp_texture = apply_lbp(image_denoised_bilateral)
        
        # Save results
        filename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(processed_images_path, f"{filename}_enhanced.png"), image_gamma)
        cv2.imwrite(os.path.join(processed_images_path, f"{filename}_edges.png"), edges)
        cv2.imwrite(os.path.join(processed_images_path, f"{filename}_gabor.png"), gabor_texture)
        cv2.imwrite(os.path.join(processed_images_path, f"{filename}_lbp.png"), lbp_texture)
        
        print(f"Processed {image_path} successfully.")
    except Exception as e:
        print(f"Failed to process {image_path} due to: {e}")

if __name__ == "__main__":
    image_folder = '/content/drive/MyDrive/Images/render/'
    processed_images_path = '/content/drive/MyDrive/Images/processed/'
    create_directory(processed_images_path)
    
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')][:30]
    for image_path in image_paths:
        process_image(image_path, processed_images_path)