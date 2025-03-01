import cv2
import numpy as np
import os

def split_image_into_patches(image_path, patch_size=256):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate number of input in each dimension
    num_patches_h = (height + patch_size - 1) // patch_size
    num_patches_w = (width + patch_size - 1) // patch_size
    
    # Create directory for input if it doesn't exist
    patches_dir = 'dataset/input'
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)
    
    patches = []
    # Extract input
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
            patch_filename = os.path.join(patches_dir, f'patch_{i}_{j}.png')
            cv2.imwrite(patch_filename, patch)
            patches.append(patch)
            
    print(f"Split image into {len(patches)} input of size {patch_size}x{patch_size}")
    return patches

if __name__ == "__main__":
    # Use the PNG file in the current directory
    image_path = "SZ2M04_L1A_01892_20240919_111839_052.png"
    patches = split_image_into_patches(image_path)
