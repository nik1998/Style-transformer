import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def gaussian_kernel(self, size=11, sigma=3):
        """Generate Gaussian kernel for structural similarity"""
        x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        x = x.view(1, -1).repeat(size, 1)
        y = x.t()
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def structural_similarity(self, x, y, kernel_size=11):
        """Calculate structural similarity between input
        Args:
            x, y: Input images (256x256)
        """
        x = torch.clamp(x, 1e-6, 1.0)
        y = torch.clamp(y, 1e-6, 1.0)

        kernel = self.gaussian_kernel(kernel_size).to(x.device)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)

        mu_x = F.conv2d(x, kernel, padding=kernel_size // 2, groups=3)
        mu_y = F.conv2d(y, kernel, padding=kernel_size // 2, groups=3)

        mu_x = torch.clamp(mu_x, min=1e-6)
        mu_y = torch.clamp(mu_y, min=1e-6)

        sigma_x = F.conv2d(x.clamp(min=0) ** 2, kernel, padding=kernel_size // 2, groups=3) - mu_x ** 2
        sigma_y = F.conv2d(y.clamp(min=0) ** 2, kernel, padding=kernel_size // 2, groups=3) - mu_y ** 2
        sigma_xy = F.conv2d((x.clamp(min=0) * y.clamp(min=0)), kernel, padding=kernel_size // 2, groups=3) - mu_x * mu_y

        sigma_x = sigma_x.clamp(min=1e-8)
        sigma_y = sigma_y.clamp(min=1e-8)

        C1 = 0.01 * torch.max(mu_x).item()
        C2 = 0.03 * torch.max(sigma_x).item()

        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        ssim = (num / (den + 1e-8)).clamp(min=-1, max=1)

        return torch.clamp(1 - ssim.mean(), min=0.0, max=1.0)

    def perceptual_loss(self, x, template):
        """Calculate perceptual loss between generated image and fog template
        Args:
            x: Generated image (256x256)
            template: Fog template (16x16)
        """
        x = x + 1e-8
        template = template + 1e-8

        # Calculate statistics for downsampled image
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_std = x.std(dim=[2, 3], keepdim=True)

        # Calculate statistics for template
        template_mean = template.mean(dim=[2, 3], keepdim=True)
        template_std = template.std(dim=[2, 3], keepdim=True)

        # Normalize both to compare patterns
        x_norm = (x - x_mean) / (x_std + 1e-8)
        template_norm = (template - template_mean) / (template_std + 1e-8)

        # Pattern similarity loss
        pattern_loss = F.mse_loss(x_norm, template_norm)

        # Statistics matching loss
        stats_loss = (F.mse_loss(x_mean, template_mean) +
                     F.mse_loss(x_std, template_std))

        return 0.5 * pattern_loss + 0.5 * stats_loss

    def forward(self, output, fogged_target, template):
        """Forward pass
        Args:
            output: Generated weather effect image (256x256)
            fogged_target: Ground truth fogged image (256x256)
            template: Fog template (16x16)
        """
        output = torch.clamp(output, 1e-8, 1.0)
        fogged_target = torch.clamp(fogged_target, 1e-8, 1.0)
        template = torch.clamp(template, 1e-8, 1.0)

        # Content loss comparing with fogged target
        l1 = self.l1_loss(output, fogged_target)

        # Structure preservation loss comparing with fogged target
        ssim = self.structural_similarity(output, fogged_target)

        # Weather pattern matching loss with template
        perceptual = self.perceptual_loss(output, template)

        # Combine losses with balanced weights
        total_loss = 0.6 * l1 + 0.3 * ssim + 0.1 * perceptual

        return total_loss, {
            'content_loss': l1.item(),
            'structure_loss': ssim.item(),
            'weather_pattern_loss': perceptual.item(),
            'total_loss': total_loss.item()
        }
