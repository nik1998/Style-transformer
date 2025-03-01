import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

from causal_transformer_model import CausalWeatherTransformer
from losses import WeatherLoss


class ClampTransform:
    """Clamp tensor values to [0,1] range"""
    def __call__(self, x):
        return torch.clamp(x, 0.0, 1.0)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class WeatherDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.split = split
        
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

        return clean_img, template_img, target_img


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
    criterion = WeatherLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Slightly lower learning rate for stability
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos'
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Best model tracking
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (clean_imgs, templates, targets) in enumerate(progress_bar):
            clean_imgs = clean_imgs.to(device)
            templates = templates.to(device)
            targets = targets.to(device)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(clean_imgs, templates)
                loss, loss_dict = criterion(outputs, targets, templates)
            
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
            avg_loss = {k: np.mean([x[k] for x in train_losses[-100:]]) for k in loss_dict.keys()}
            progress_bar.set_postfix(avg_loss)
        
        # Validation loop
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for clean_imgs, templates, targets in val_loader:
                clean_imgs = clean_imgs.to(device)
                templates = templates.to(device)
                targets = targets.to(device)
                
                outputs = model(clean_imgs, templates)
                loss, loss_dict = criterion(outputs, targets, templates)
                val_losses.append(loss_dict)
        
        # Calculate average validation loss
        avg_val_loss = np.mean([x['total_loss'] for x in val_losses])
        
        # Print validation metrics
        print("\nValidation metrics:")
        for k in val_losses[0].keys():
            avg = np.mean([x[k] for x in val_losses])
            print(f"{k}: {avg:.4f}")
        print()
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
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
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize datasets
    train_dataset = WeatherDataset('/home/nik/workprojects/PHD/dataset/training', split='train')
    val_dataset = WeatherDataset('/home/nik/workprojects/PHD/dataset/training', split='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
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
