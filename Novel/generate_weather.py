import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from torchvision import transforms
import json
from pathlib import Path

from causal_transformer_model import CausalWeatherTransformer

class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.test_dir = self.data_dir / 'test'
        
        # Load relationships file
        with open(self.data_dir / 'relationships.json', 'r') as f:
            relationships = json.load(f)
        self.relationships = relationships['test']
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.relationships)

    def __getitem__(self, idx):
        rel = self.relationships[idx]
        
        # Load clean image
        clean_path = self.test_dir / 'clean' / rel['clean']
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Load template image
        template_path = self.test_dir / 'template' / rel['template']
        template_img = Image.open(template_path).convert('RGB')
        
        # Load target for comparison
        target_path = self.test_dir / 'target' / rel['target']
        target_img = Image.open(target_path).convert('RGB')
        
        if self.transform:
            clean_img = self.transform(clean_img)
            template_img = self.transform(template_img)
            target_img = self.transform(target_img)

        return clean_img, template_img, target_img, rel['clean']

def calculate_metrics(generated, target):
    """Calculate PSNR and SSIM between generated and target images"""
    mse = F.mse_loss(generated, target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # Simple SSIM
    mu_x = F.avg_pool2d(generated, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
    
    sigma_x = F.avg_pool2d(generated**2, kernel_size=11, stride=1, padding=5) - mu_x**2
    sigma_y = F.avg_pool2d(target**2, kernel_size=11, stride=1, padding=5) - mu_y**2
    sigma_xy = F.avg_pool2d(generated*target, kernel_size=11, stride=1, padding=5) - mu_x*mu_y
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
    return psnr.item(), ssim.mean().item()

def generate_weather_images():
    """Generate weather-affected images using test dataset samples"""
    # Parameters
    model_path = 'checkpoints/best_model.pth'
    data_dir = '../dataset/training'  # Directory containing train/test splits
    output_dir = 'weather_output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/generated', exist_ok=True)
    os.makedirs(output_dir + '/clean', exist_ok=True)
    os.makedirs(output_dir + '/target', exist_ok=True)
    os.makedirs(output_dir + '/template', exist_ok=True)
    
    # Initialize model and load weights
    model = CausalWeatherTransformer().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize dataset and dataloader
    dataset = TestDataset(data_dir)
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
            for output, clean, template, target, filename in zip(outputs, clean_imgs, templates, targets, filenames):
                # Convert to numpy and scale to 0-255
                output_img = (output.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                clean_img = (clean.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                template_img = (template.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                target_img = (target.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Save all versions
                Image.fromarray(output_img).save(os.path.join(output_dir, 'generated', filename))
                Image.fromarray(clean_img).save(os.path.join(output_dir, 'clean', filename))
                Image.fromarray(template_img).save(os.path.join(output_dir, 'template', filename))
                Image.fromarray(target_img).save(os.path.join(output_dir, 'target', filename))
    
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
