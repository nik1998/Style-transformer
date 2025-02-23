import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from torchvision import transforms

from causal_transformer_model import CausalWeatherTransformer

class ImageDataset(Dataset):
    def __init__(self, image_dir, template_path):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Load template
        template = Image.open(template_path).convert('RGB')
        self.template = self.transform(template)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, self.template, img_name

def generate_weather_images():
    """
    Generate weather-affected images from input directory with hardcoded parameters
    """
    # Hardcoded parameters
    model_path = 'checkpoints/best_model.pth'
    input_dir = '../dataset/input'
    output_dir = 'weather_output'
    template_path = '../dataset/fog_template.png'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and load weights
    model = CausalWeatherTransformer().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize dataset and dataloader
    dataset = ImageDataset(input_dir, template_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    with torch.no_grad():
        for images, templates, filenames in tqdm(dataloader, desc="Generating weather effects"):
            images = images.to(device)
            templates = templates.to(device)
            
            # Generate weather effects
            outputs = model(images, templates)
            
            # Save each image in the batch
            for output, filename in zip(outputs, filenames):
                # Convert to numpy and scale to 0-255
                output_img = (output.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Save with original filename using PIL to preserve RGB order
                output_path = os.path.join(output_dir, filename)
                Image.fromarray(output_img).save(output_path)

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate weather-affected images
    generate_weather_images()
