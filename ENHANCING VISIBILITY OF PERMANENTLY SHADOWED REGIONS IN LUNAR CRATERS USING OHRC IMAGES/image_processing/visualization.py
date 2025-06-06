import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_image(image, title, cmap='gray'):
    plt.figure(figsize=(12, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_comparison(original_image, processed_image, title1, title2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.show()

def show_histogram(image, title):
    plt.figure(figsize=(8, 4))
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def overlay_edges_on_image(original_image, edges):
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR), 0.8, edges_colored, 0.2, 0)