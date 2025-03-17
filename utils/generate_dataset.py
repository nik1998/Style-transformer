import json
import os
import random
import shutil

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from utils.fog_generator import FogGenerator
from utils.rain_generator import RainGenerator
from utils.snow_generator import SnowGenerator


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


EFFECT_TYPES = ["fog", "rain", "snow"]
#EFFECT_TYPES = ["fog"]


def split_image_into_patches(image_path, patch_size=256):
    """Split an image into patches without padding"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    height, width = img.shape[:2]

    # Calculate number of patches in each dimension
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    patches = []
    patch_info = []

    # Extract patches
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Calculate patch coordinates
            start_h = i * patch_size
            start_w = j * patch_size
            end_h = min(start_h + patch_size, height)
            end_w = min(start_w + patch_size, width)

            # Skip if patch is smaller than patch_size
            if end_h - start_h < patch_size or end_w - start_w < patch_size:
                continue

            # Extract patch
            patch = img[start_h:end_h, start_w:end_w]
            patches.append(patch)
            patch_info.append((i, j))

    return patches, patch_info


def apply_weather_effect(image, effect_type, seed):
    """Apply a specific weather effect to an image"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set the random seed for consistent results
    random.seed(seed)
    np.random.seed(seed)

    if effect_type == "fog":
        generator = FogGenerator(seed=seed)
        result = generator.apply_noise(image_rgb)
    elif effect_type == "rain":
        generator = RainGenerator(seed=seed)
        # Use a fixed algorithm with the seed instead of random.randint
        algorithm = (seed % 3) + 1  # 1, 2, or 3
        result = generator.apply_rain(image_rgb, algorithm=algorithm, seed=seed)
    elif effect_type == "snow":
        generator = SnowGenerator(seed=seed)
        # Use a fixed algorithm with the seed instead of random.randint
        algorithm = (seed % 2) + 1  # 1 or 2
        result = generator.apply_snow(image_rgb, algorithm=algorithm, seed=seed)
    else:
        raise ValueError(f"Unknown effect type: {effect_type}")

    # Convert back to BGR
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def process_image(input_path, template_paths, output_base_dir, split_type, idx, effect_type=None):
    """Process a single image with multiple templates and weather effects"""
    try:
        # Read input image
        input_image = cv2.imread(input_path)
        if input_image is None:
            print(f"Could not read image: {input_path}")
            return None

        # Create split-specific directories
        split_dir = os.path.join(output_base_dir, split_type)
        clean_dir = os.path.join(split_dir, "clean")
        template_dir = os.path.join(split_dir, "template")
        clean_template_dir = os.path.join(split_dir, "clean_template")
        target_dir = os.path.join(split_dir, "target")

        for d in [clean_dir, template_dir, clean_template_dir, target_dir]:
            ensure_dir(d)

        # Save clean input image
        clean_name = f"clean_{idx}.png"
        clean_path = os.path.join(clean_dir, clean_name)
        cv2.imwrite(clean_path, input_image)

        relationships = []

        # Process each template
        for template_idx, template_path in enumerate(template_paths):
            try:
                # Read template image
                template_image = cv2.imread(template_path)
                if template_image is None:
                    print(f"Could not read template: {template_path}")
                    continue

                # If no effect type is specified, randomly choose one
                if effect_type is None:
                    current_effect = random.choice(EFFECT_TYPES)
                else:
                    current_effect = effect_type

                # Generate noise seed for this template
                noise_seed = random.randint(0, 1000000)

                # Save clean template
                clean_template_name = f"clean_template_{idx}_{template_idx}.png"
                clean_template_save_path = os.path.join(clean_template_dir, clean_template_name)
                cv2.imwrite(clean_template_save_path, template_image)
                
                # Apply same weather effect to both template and input image
                noisy_template = apply_weather_effect(template_image, current_effect, noise_seed)
                target = apply_weather_effect(input_image, current_effect, noise_seed)

                # Save template and target
                template_name = f"template_{idx}_{template_idx}_{current_effect}.png"
                target_name = f"target_{idx}_{template_idx}_{current_effect}.png"

                template_save_path = os.path.join(template_dir, template_name)
                target_save_path = os.path.join(target_dir, target_name)

                cv2.imwrite(template_save_path, noisy_template)
                cv2.imwrite(target_save_path, target)

                # Record relationship
                relationships.append({
                    "clean": clean_name,
                    "clean_template": clean_template_name,
                    "template": template_name,
                    "target": target_name,
                    "original_template": os.path.basename(template_path),
                    "noise_seed": noise_seed,
                    "effect_type": current_effect
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
    if h < patch_size[0] or w < patch_size[1]:
        return None

    top = random.randint(0, h - patch_size[0])
    left = random.randint(0, w - patch_size[1])
    patch = image[top:top + patch_size[0], left:left + patch_size[1]]
    return patch


def generate_dataset(input_dir="../examples/images",
                     output_dir="../dataset/training",
                     test_size=0.2,
                     templates_per_image=3,
                     patch_size=256):
    """Generate dataset with train/test split and multiple templates per image with weather effects"""
    try:
        # Get list of input images
        input_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not input_images:
            print("No input images found!")
            return

        print(f"Found {len(input_images)} input images")

        # Create temporary directory for patches
        temp_patches_dir = os.path.join(output_dir, "temp_patches")
        ensure_dir(temp_patches_dir)

        # Split all images into patches and save them
        all_patches = []
        for img_path in input_images:
            print(f"Splitting image {img_path} into patches...")
            patches, patch_info = split_image_into_patches(img_path, patch_size)

            for i, (patch, (row, col)) in enumerate(zip(patches, patch_info)):
                patch_path = os.path.join(temp_patches_dir,
                                          f"{os.path.basename(img_path).split('.')[0]}_patch_{row}_{col}.png")
                cv2.imwrite(patch_path, patch)
                all_patches.append(patch_path)

        print(f"Created {len(all_patches)} patches")

        # Split patches into train/test
        train_patches, test_patches = train_test_split(all_patches, test_size=test_size, random_state=42)

        # Clean up existing output directory if it exists (except temp_patches)
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                if item != "temp_patches":
                    item_path = os.path.join(output_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)

        # Create directory structure
        for split in ['train', 'test']:
            split_dir = os.path.join(output_dir, split)
            ensure_dir(split_dir)
            for subdir in ['clean', 'template', 'clean_template', 'target']:
                ensure_dir(os.path.join(split_dir, subdir))

        relationships = {'train': [], 'test': []}

        # Process training patches
        print("\nProcessing training patches...")
        for idx, patch_path in enumerate(train_patches):
            # Select other patches to use as templates
            other_patches = [p for p in all_patches if p != patch_path]
            template_patches = random.sample(other_patches, min(templates_per_image, len(other_patches)))

            # Process each weather effect type
            for effect_type in EFFECT_TYPES:
                rels = process_image(patch_path,
                                     template_patches,
                                     output_dir,
                                     'train',
                                     f"{idx}_{effect_type}",
                                     effect_type)

                if rels:
                    relationships['train'].extend(rels)

            print(f"Processed training patch {idx + 1}/{len(train_patches)}: {patch_path}")

        # Process test patches
        print("\nProcessing test patches...")
        for idx, patch_path in enumerate(test_patches):
            # Select other patches to use as templates
            other_patches = [p for p in all_patches if p != patch_path]
            template_patches = random.sample(other_patches, min(templates_per_image, len(other_patches)))

            # Process each weather effect type
            for effect_type in EFFECT_TYPES:
                rels = process_image(patch_path,
                                     template_patches,
                                     output_dir,
                                     'test',
                                     f"{idx}_{effect_type}",
                                     effect_type)

                if rels:
                    relationships['test'].extend(rels)

            print(f"Processed test patch {idx + 1}/{len(test_patches)}: {patch_path}")

        # Save relationships to JSON file
        relationships_file = os.path.join(output_dir, 'relationships.json')
        with open(relationships_file, 'w') as f:
            json.dump(relationships, f, indent=2)

        # Clean up temporary patches directory
        shutil.rmtree(temp_patches_dir)

        print("\nDataset generation complete!")
        print(f"Train set: {len(train_patches)} patches with fog, rain, and snow effects")
        print(f"Test set: {len(test_patches)} patches with fog, rain, and snow effects")
        print(f"Relationships saved to {relationships_file}")

    except Exception as e:
        print(f"Error generating dataset: {str(e)}")


if __name__ == "__main__":
    # Generate dataset
    generate_dataset(templates_per_image=3)  # Generate 3 templates per input image
