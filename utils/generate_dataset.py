import os
import cv2
import numpy as np
from PIL import Image
import random
import json
import shutil
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_perlin_noise(shape, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """Generate Perlin noise for more natural fog patterns"""
    if seed is not None:
        np.random.seed(seed)
        
    def generate_octave(shape, frequency):
        noise = np.random.rand(*shape)
        noise = cv2.resize(noise, None, fx=frequency, fy=frequency, interpolation=cv2.INTER_LINEAR)
        noise = cv2.resize(noise, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return noise
    
    noise = np.zeros(shape)
    amplitude = 1.0
    frequency = 1.0 / scale
    
    for _ in range(octaves):
        noise += amplitude * generate_octave(shape, frequency)
        amplitude *= persistence
        frequency *= lacunarity
        
    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

def simulate_atmospheric_scattering(image, fog_density, depth_factor=1.0):
    """Simulate atmospheric scattering effect in fog"""
    # Convert to float
    image_float = image.astype(np.float32) / 255.0
    
    # Create depth map (simple gradient from top to bottom)
    height = image.shape[0]
    depth_map = np.linspace(0, 1, height)
    depth_map = np.tile(depth_map.reshape(-1, 1), (1, image.shape[1]))
    depth_map = np.dstack([depth_map] * 3)
    
    # Apply depth-dependent scattering
    transmission = np.exp(-fog_density * depth_map * depth_factor)
    airlight = np.array([1.0, 1.0, 1.0])  # White fog
    
    scattered = image_float * transmission + airlight.reshape(1, 1, 3) * (1 - transmission)
    return np.clip(scattered, 0, 1)

def apply_noise(image, seed=None):
    """Apply realistic fog effect to an image with optional seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to float32
    image_float = image.astype(np.float32) / 255.0
    
    # Use seed to initialize random state for consistent parameters
    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState()
        
    # Generate base perlin noise for fog pattern with consistent parameters
    base_scale = random_state.uniform(50, 150)
    base_octaves = random_state.randint(4, 9)
    base_persistence = random_state.uniform(0.4, 0.6)
    base_lacunarity = random_state.uniform(1.8, 2.2)
    
    fog_pattern = generate_perlin_noise(
        image.shape[:2], 
        scale=base_scale,
        octaves=base_octaves,
        persistence=base_persistence,
        lacunarity=base_lacunarity,
        seed=seed if seed is not None else random_state.randint(0, 1000000)
    )
    
    # Add multi-scale detail with same seed
    detail_scale = random_state.uniform(20, 50)
    detail_pattern = generate_perlin_noise(
        image.shape[:2],
        scale=detail_scale,
        octaves=3,
        persistence=0.3,
        seed=seed if seed is not None else random_state.randint(0, 1000000)
    )
    
    fog_pattern = 0.8 * fog_pattern + 0.2 * detail_pattern
    
    # Use same random state for blur
    blur_sigma = random_state.uniform(3, 5)  # Increased blur for thicker fog
    fog_pattern = gaussian_filter(fog_pattern, sigma=blur_sigma)
    
    # Expand to 3 channels
    fog_pattern = np.stack([fog_pattern] * 3, axis=-1)
    
    # Use same random state for color variation
    fog_color = np.array([
        random_state.uniform(0.97, 1.0),  # R - increased brightness
        random_state.uniform(0.97, 1.0),  # G
        random_state.uniform(0.97, 1.0)   # B
    ])
    fog = fog_pattern * fog_color.reshape(1, 1, 3)
    
    # Use same random state for fog parameters
    fog_intensity = random_state.uniform(0.7, 0.95)
    fog_density = random_state.uniform(0.7, 0.9)
    
    # Apply atmospheric scattering with consistent density
    scattered = simulate_atmospheric_scattering(
        image, 
        fog_density=fog_density
    )
    
    # Combine effects with stronger fog influence
    noisy_image = scattered * (1 - fog_intensity * fog_pattern) + fog * fog_intensity
    
    # Add subtle gaussian noise with consistent parameters
    noise_std = random_state.uniform(0.01, 0.03)
    noise = random_state.normal(0, noise_std, noisy_image.shape)  # Reduced noise for cleaner fog
    noisy_image = noisy_image + noise
    
    # Clip values to valid range
    noisy_image = np.clip(noisy_image, 0, 1)
    
    # Convert back to uint8
    return (noisy_image * 255).astype(np.uint8)

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
                
                # Apply same noise pattern to both template and input image
                noisy_template = apply_noise(template_image, seed=noise_seed)
                target = apply_noise(input_image, seed=noise_seed)
                
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
