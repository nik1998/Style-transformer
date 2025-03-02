import os
import cv2
import numpy as np
from PIL import Image
import random
import json
import shutil
from sklearn.model_selection import train_test_split
from utils.fog_generator import FogGenerator

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_image(input_path, template_paths, output_base_dir, split_type, idx):
    """Process a single image with multiple templates"""
    try:
        # Read input image
        input_image = cv2.imread(input_path)
        if input_image is None:
            print(f"Could not read image: {input_path}")
            return None
        
        # Convert BGR to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Create split-specific directories
        split_dir = os.path.join(output_base_dir, split_type)
        clean_dir = os.path.join(split_dir, "clean")
        template_dir = os.path.join(split_dir, "template")
        target_dir = os.path.join(split_dir, "target")
        
        for d in [clean_dir, template_dir, target_dir]:
            ensure_dir(d)
        
        # Save clean input image
        clean_name = f"clean_{idx}.png"
        clean_path = os.path.join(clean_dir, clean_name)
        cv2.imwrite(clean_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
        
        relationships = []
        
        # Process each template
        for template_idx, template_path in enumerate(template_paths):
            try:
                # Read template image
                template_image = cv2.imread(template_path)
                if template_image is None:
                    print(f"Could not read template: {template_path}")
                    continue
                    
                template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
                
                # Generate noise seed for this template
                noise_seed = random.randint(0, 1000000)
                
                # Create fog generator with the seed
                fog_generator = FogGenerator(seed=noise_seed)
                
                # Apply same noise pattern to both template and input image
                noisy_template = fog_generator.apply_noise(template_image)
                target = fog_generator.apply_noise(input_image)
                
                # Save template and target
                template_name = f"template_{idx}_{template_idx}.png"
                target_name = f"target_{idx}_{template_idx}.png"
                
                template_save_path = os.path.join(template_dir, template_name)
                target_save_path = os.path.join(target_dir, target_name)
                
                cv2.imwrite(template_save_path, cv2.cvtColor(noisy_template, cv2.COLOR_RGB2BGR))
                cv2.imwrite(target_save_path, cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
                
                # Record relationship
                relationships.append({
                    "clean": clean_name,
                    "template": template_name,
                    "target": target_name,
                    "original_template": os.path.basename(template_path),
                    "noise_seed": noise_seed
                })
                
            except Exception as e:
                print(f"Error processing template {template_path}: {str(e)}")
                continue
        
        return relationships
        
    except Exception as e:
        print(f"Error processing input image {input_path}: {str(e)}")
        return None

def extract_random_patch(image, patch_size=(256, 256)):
    """Extract a random patch from an image"""
    h, w = image.shape[:2]
    top = random.randint(0, h - patch_size[0])
    left = random.randint(0, w - patch_size[1])
    patch = image[top:top + patch_size[0], left:left + patch_size[1]]
    return patch

def generate_dataset(input_dir="dataset/input", 
                    output_dir="dataset/training",
                    test_size=0.2,
                    templates_per_image=3,
                    patch_size=(256, 256)):
    """Generate dataset with train/test split and multiple templates per image"""
    try:
        # Get list of input images
        input_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not input_images:
            print("No input images found!")
            return
            
        print(f"Found {len(input_images)} input images")
        
        # Split input images into train/test
        train_images, test_images = train_test_split(input_images, test_size=test_size, random_state=42)
        
        # Clean up existing output directory if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # Create directory structure
        for split in ['train', 'test']:
            split_dir = os.path.join(output_dir, split)
            ensure_dir(split_dir)
            for subdir in ['clean', 'template', 'target']:
                ensure_dir(os.path.join(split_dir, subdir))
        
        relationships = {'train': [], 'test': []}
        
        # Process training images
        print("\nProcessing training images...")
        for idx, image in enumerate(train_images):
            # Read the current image
            current_image = cv2.imread(os.path.join(input_dir, image))
            if current_image is None:
                print(f"Could not read image: {image}")
                continue
                
            # Select other images to use as templates
            other_images = [img for img in input_images if img != image]
            template_images = random.sample(other_images, min(templates_per_image, len(other_images)))
            
            # Extract random patches from template images
            template_patches = []
            for template_img in template_images:
                img = cv2.imread(os.path.join(input_dir, template_img))
                if img is not None:
                    patch = extract_random_patch(img, patch_size)
                    template_patches.append(patch)
            
            # Save template patches to temporary files
            temp_template_paths = []
            for i, patch in enumerate(template_patches):
                temp_path = os.path.join(output_dir, f'temp_template_{idx}_{i}.png')
                cv2.imwrite(temp_path, patch)
                temp_template_paths.append(temp_path)
            
            # Process image with template patches
            rels = process_image(os.path.join(input_dir, image),
                               temp_template_paths,
                               output_dir,
                               'train',
                               idx)
            
            # Clean up temporary template files
            for temp_path in temp_template_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            if rels:
                relationships['train'].extend(rels)
            print(f"Processed training image {idx + 1}/{len(train_images)}: {image}")
        
        # Process test images
        print("\nProcessing test images...")
        for idx, image in enumerate(test_images):
            # Read the current image
            current_image = cv2.imread(os.path.join(input_dir, image))
            if current_image is None:
                print(f"Could not read image: {image}")
                continue
                
            # Select other images to use as templates
            other_images = [img for img in input_images if img != image]
            template_images = random.sample(other_images, min(templates_per_image, len(other_images)))
            
            # Extract random patches from template images
            template_patches = []
            for template_img in template_images:
                img = cv2.imread(os.path.join(input_dir, template_img))
                if img is not None:
                    patch = extract_random_patch(img, patch_size)
                    template_patches.append(patch)
            
            # Save template patches to temporary files
            temp_template_paths = []
            for i, patch in enumerate(template_patches):
                temp_path = os.path.join(output_dir, f'temp_template_{idx}_{i}.png')
                cv2.imwrite(temp_path, patch)
                temp_template_paths.append(temp_path)
            
            # Process image with template patches
            rels = process_image(os.path.join(input_dir, image),
                               temp_template_paths,
                               output_dir,
                               'test',
                               idx)
            
            # Clean up temporary template files
            for temp_path in temp_template_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            if rels:
                relationships['test'].extend(rels)
            print(f"Processed test image {idx + 1}/{len(test_images)}: {image}")
        
        # Save relationships to JSON file
        relationships_file = os.path.join(output_dir, 'relationships.json')
        with open(relationships_file, 'w') as f:
            json.dump(relationships, f, indent=2)
        
        print("\nDataset generation complete!")
        print(f"Train set: {len(train_images)} input images")
        print(f"Test set: {len(test_images)} input images")
        print(f"Relationships saved to {relationships_file}")
        
    except Exception as e:
        print(f"Error generating dataset: {str(e)}")

if __name__ == "__main__":
    # Generate dataset
    generate_dataset(templates_per_image=3)  # Generate 3 templates per input image
