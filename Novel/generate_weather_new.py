import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from PIL import Image

from transformer_model import WeatherTransformer
from datasets import WeatherDataset, calculate_metrics, save_images

def generate_weather_images():
    """Generate weather-affected images using test dataset samples"""
    # Parameters
    model_path = 'checkpoints/best_model.pth'
    data_dir = '../dataset/training'  # Directory containing train/test splits
    output_dir = 'weather_output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and load weights
    model = WeatherTransformer().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize dataset and dataloader
    dataset = WeatherDataset(data_dir, split='test', return_filename=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    metrics = []
    with torch.no_grad():
        for clean_imgs, templates, targets, filenames in tqdm(dataloader, desc="Generating weather effects"):
            clean_imgs = clean_imgs.to(device)
            templates = templates.to(device)
            targets = targets.to(device)
            
            # Generate weather effects
            outputs = model(clean_imgs, templates)
            
            # Calculate metrics
            for output, target in zip(outputs, targets):
                psnr, ssim = calculate_metrics(output, target)
                metrics.append({'psnr': psnr, 'ssim': ssim})
            
            # Save images
            save_images(outputs, clean_imgs, templates, targets, filenames, output_dir)
    
    # Calculate and print average metrics
    avg_psnr = np.mean([m['psnr'] for m in metrics])
    avg_ssim = np.mean([m['ssim'] for m in metrics])
    print(f"\nAverage PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.2f}")
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'metrics': metrics,
            'average': {
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }
        }, f, indent=2)

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate weather-affected images
    generate_weather_images()
