#!/usr/bin/env python3
import os
import sys
import requests
import random
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import time

# Import the split_image_into_patches function from image_patches.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_patches import split_image_into_patches

# Create directories if they don't exist
examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
images_dir = os.path.join(examples_dir, 'images')
patches_dir = os.path.join(examples_dir, 'patches')

os.makedirs(images_dir, exist_ok=True)
os.makedirs(patches_dir, exist_ok=True)

# Function to download an image from a URL
def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        # Verify the image can be opened
        img = cv2.imread(save_path)
        if img is None:
            print(f"Failed to read downloaded image from {url}")
            return False
        
        print(f"Successfully downloaded and saved image to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return False

# Function to split image into patches and save to patches directory
def process_image(image_path, patch_size=256):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate number of patches in each dimension
    num_patches_h = (height + patch_size - 1) // patch_size
    num_patches_w = (width + patch_size - 1) // patch_size
    
    patches_count = 0
    # Extract patches
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Calculate patch coordinates
            start_h = i * patch_size
            start_w = j * patch_size
            end_h = min(start_h + patch_size, height)
            end_w = min(start_w + patch_size, width)
            
            # Extract patch
            patch = img[start_h:end_h, start_w:end_w]
            
            # If patch is smaller than patch_size, pad it
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded_patch
            
            # Save patch
            image_name = os.path.basename(image_path).split('.')[0]
            patch_filename = os.path.join(patches_dir, f'{image_name}_patch_{i}_{j}.png')
            cv2.imwrite(patch_filename, patch)
            patches_count += 1
            
    print(f"Split image {image_path} into {patches_count} patches of size {patch_size}x{patch_size}")
    return patches_count

# List of image URLs for city streets and nature without precipitation
image_urls = [
    # City streets without precipitation
    "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80",  # City street
    "https://images.unsplash.com/photo-1444723121867-7a241cacace9?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80",  # City view
    "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80",  # Urban street
    
    # Nature without precipitation
    "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80",  # Forest
    "https://images.unsplash.com/photo-1501854140801-50d01698950b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80",  # Mountain landscape
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80",  # Forest path
]

def main():
    print("Starting image download and processing...")
    
    # Download images
    downloaded_images = []
    for i, url in enumerate(image_urls):
        image_path = os.path.join(images_dir, f"image_{i}.jpg")
        if download_image(url, image_path):
            downloaded_images.append(image_path)
    
    if not downloaded_images:
        print("No images were successfully downloaded. Exiting.")
        return
    
    print(f"Successfully downloaded {len(downloaded_images)} images.")
    
    # Process images
    total_patches = 0
    for image_path in downloaded_images:
        try:
            patches_count = process_image(image_path)
            total_patches += patches_count
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    print(f"Finished processing. Created a total of {total_patches} patches.")

if __name__ == "__main__":
    main()
