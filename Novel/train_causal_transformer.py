import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm
import numpy as np
from pathlib import Path

from causal_transformer_model import CausalWeatherTransformer


class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # Use VGG16 features for perceptual loss
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device)
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ]).eval()
        
        # Freeze VGG parameters
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x, y):
        x_features = []
        y_features = []
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            x_features.append(x)
            y_features.append(y)
            
        perceptual_loss = 0.0
        for x_feat, y_feat in zip(x_features, y_features):
            perceptual_loss += F.mse_loss(x_feat, y_feat)
            
        return perceptual_loss


class WeatherDataset(Dataset):
    def __init__(self, image_dir, template_path, transform=None, split='train'):
        self.image_dir = image_dir
        self.split = split
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if split == 'train':
            self.image_files = self.image_files[:int(0.9 * len(self.image_files))]
        else:
            self.image_files = self.image_files[int(0.9 * len(self.image_files)):]
            
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Load template (keep on CPU)
        template = Image.open(template_path).convert('RGB')
        self.template = self.transform(template)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load clean image
        clean_path = os.path.join(self.image_dir, img_name)
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Load corresponding weather-affected image
        weather_path = os.path.join('../dataset/fog_images2', f'fog_{img_name}')
        weather_img = Image.open(weather_path).convert('RGB')

        if self.transform:
            clean_img = self.transform(clean_img)
            weather_img = self.transform(weather_img)

        return clean_img, self.template, weather_img


class WeatherLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device=device).to(device)
        self.edge_loss = self.edge_detection_loss
        
    def edge_detection_loss(self, x, y):
        def sobel_edges(img):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device).float()
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device).float()
            
            sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            
            edge_x = F.conv2d(img, sobel_x, padding=1, groups=3)
            edge_y = F.conv2d(img, sobel_y, padding=1, groups=3)
            
            return torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-6)
            
        return F.l1_loss(sobel_edges(x), sobel_edges(y))
    
    def checkerboard_loss(self, x):
        """Detect and penalize checkerboard patterns"""
        # Compute differences between adjacent pixels
        diff_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        # Detect alternating patterns
        checker_x = torch.abs(diff_x[:, :, :, 1:] - diff_x[:, :, :, :-1])
        checker_y = torch.abs(diff_y[:, :, 1:, :] - diff_y[:, :, :-1, :])
        
        # High values indicate presence of checkerboard artifacts
        return checker_x.mean() + checker_y.mean()
    
    def smoothness_loss(self, x):
        """Encourage local smoothness while preserving edges"""
        # Compute gradients
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        # Compute second-order gradients
        grad_xx = torch.abs(grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1])
        grad_yy = torch.abs(grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :])
        
        return grad_xx.mean() + grad_yy.mean()
    
    def forward(self, output, target, template):
        # Content loss
        l1 = self.l1_loss(output, target)
        
        # Perceptual loss
        perceptual = self.perceptual_loss(output, target)
        
        # Edge preservation loss
        edge = self.edge_loss(output, target)
        
        # Anti-checkerboard losses
        checker = self.checkerboard_loss(output)
        smooth = self.smoothness_loss(output)
        
        # Weather pattern consistency
        pattern = self.l1_loss(
            F.adaptive_avg_pool2d(output, (32, 32)),
            F.adaptive_avg_pool2d(template, (32, 32))
        )
        
        # Total loss with weights
        total_loss = (
            0.4 * l1 +           # Content preservation
            0.2 * perceptual +   # Perceptual quality
            0.15 * edge +        # Structure preservation
            0.1 * pattern +      # Weather pattern matching
            0.1 * checker +      # Anti-checkerboard
            0.05 * smooth        # Local smoothness
        )
        
        return total_loss, {
            'l1_loss': l1.item(),
            'perceptual_loss': perceptual.item(),
            'edge_loss': edge.item(),
            'pattern_loss': pattern.item(),
            'checker_loss': checker.item(),
            'smooth_loss': smooth.item(),
            'total_loss': total_loss.item()
        }


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    device='cuda',
    save_dir='checkpoints'
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup training components
    criterion = WeatherLoss(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos'
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Best model tracking
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (clean_imgs, templates, weather_imgs) in enumerate(progress_bar):
            clean_imgs = clean_imgs.to(device)
            templates = templates.to(device)
            weather_imgs = weather_imgs.to(device)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(clean_imgs, templates)
                loss, loss_dict = criterion(outputs, weather_imgs, templates)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update progress bar
            train_losses.append(loss_dict)
            progress_bar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
            
            # Print current loss
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: {loss_dict}")
        
        # Validation loop
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for clean_imgs, templates, weather_imgs in val_loader:
                clean_imgs = clean_imgs.to(device)
                templates = templates.to(device)
                weather_imgs = weather_imgs.to(device)
                
                outputs = model(clean_imgs, templates)
                loss, loss_dict = criterion(outputs, weather_imgs, templates)
                val_losses.append(loss_dict)
        
        # Calculate average validation loss
        avg_val_loss = np.mean([x['total_loss'] for x in val_losses])
        # Print validation metrics
        print("\nValidation metrics:")
        for k in val_losses[0].keys():
            print(f"{k}: {np.mean([x[k] for x in val_losses]):.4f}")
        print()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))


if __name__ == "__main__":
    # Set multiprocessing start method to spawn for CUDA support
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create necessary directories
    os.makedirs('../dataset/input', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize datasets
    train_dataset = WeatherDataset('../dataset/input', '../fog_template.png', split='train')
    val_dataset = WeatherDataset('../dataset/input', '../fog_template.png', split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,  # Reduced number of workers for better CUDA stability
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,  # Reduced number of workers for better CUDA stability
        pin_memory=True
    )
    
    # Initialize model
    model = CausalWeatherTransformer().to(device)
    
    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        device=device,
        save_dir='checkpoints'
    )
