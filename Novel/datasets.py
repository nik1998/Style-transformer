import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import numpy as np


class ClampTransform:
    """Clamp tensor values to [0,1] range"""
    def __call__(self, x):
        return torch.clamp(x, 0.0, 1.0)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class WeatherDataset(Dataset):
    """Dataset for training, validation, and testing with weather effect images"""
    def __init__(self, data_dir, split='train', transform=None, return_filename=False):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.return_filename = return_filename
        
        # Load relationships file
        with open(Path(data_dir) / 'relationships.json', 'r') as f:
            relationships = json.load(f)
        self.relationships = relationships[split]
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            ClampTransform()  # Ensure input values are in [0,1]
        ])

    def __len__(self):
        return len(self.relationships)

    def __getitem__(self, idx):
        rel = self.relationships[idx]
        
        # Load clean image
        clean_path = self.data_dir / 'clean' / rel['clean']
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Load template image
        template_path = self.data_dir / 'template' / rel['template']
        template_img = Image.open(template_path).convert('RGB')
        
        # Load target (weather-affected) image
        target_path = self.data_dir / 'target' / rel['target']
        target_img = Image.open(target_path).convert('RGB')

        if self.transform:
            clean_img = self.transform(clean_img)
            template_img = self.transform(template_img)
            target_img = self.transform(target_img)

        if self.return_filename:
            return clean_img, template_img, target_img, rel['clean']
        else:
            return clean_img, template_img, target_img


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


def save_images(outputs, clean_imgs, templates, targets, filenames, output_dir):
    """Save generated images and their corresponding inputs/targets"""
    import os
    from PIL import Image
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'generated'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'clean'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'template'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'target'), exist_ok=True)
    
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
