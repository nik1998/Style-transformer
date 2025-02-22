import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

        self.conv_layers = nn.Sequential(
            # Patch embedding with strided convolution
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            # Local feature processing
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Apply convolutions
        x = self.conv_layers(x)

        # Reshape to sequence format
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x.reshape(B, H * W, C)  # [B, N, C]

        # Apply normalization
        x = self.norm(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Efficient projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.norm(x)

        # Project queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # Compute attention scores with memory-efficient implementation
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Create causal mask
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, N, D]

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.dropout(self.out_proj(out))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP with reduced complexity
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvolutionalDecoder(nn.Module):
    def __init__(self, embed_dim=256, out_channels=3):
        super().__init__()

        # Gradual channel reduction with spatial smoothing
        self.conv_layers = nn.ModuleList([
            # First stage
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU()
            ),
            # Second stage
            nn.Sequential(
                nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim // 4),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim // 4),
                nn.GELU()
            ),
            # Final refinement
            nn.Sequential(
                nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim // 8),
                nn.GELU(),
                nn.Conv2d(embed_dim // 8, out_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        ])

        # Smooth upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Calculate required upsampling steps
        input_size = x.shape[-1]
        target_size = 256
        num_upsample_needed = max(0, int(np.log2(target_size // input_size)))

        # Apply progressive refinement with upsampling
        for i, conv_block in enumerate(self.conv_layers[:-1]):
            x = conv_block(x)
            if i < num_upsample_needed:  # Only upsample if needed
                x = self.upsample(x)

        # Final refinement
        x = self.conv_layers[-1](x)

        # Ensure output is correct size
        if x.shape[-1] != target_size:
            x = F.interpolate(x, size=(target_size, target_size),
                              mode='bilinear', align_corners=True)
        return x


class CausalWeatherTransformer(nn.Module):
    def __init__(self,
                 image_size=256,
                 in_channels=3,
                 embed_dim=256,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 patch_size=16):
        super().__init__()

        self.image_size = image_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Patch-based encoders
        self.encoder = ConvolutionalEncoder(in_channels, embed_dim, patch_size)
        self.template_encoder = ConvolutionalEncoder(in_channels, embed_dim, patch_size)

        # Position embeddings
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.template_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Output processing
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = ConvolutionalDecoder(embed_dim, in_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.template_token, std=0.02)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, template):
        B = img.shape[0]

        # Encode input image and template
        x = self.encoder(img)
        template_embed = self.template_encoder(template)

        # Process template token
        template_tokens = self.template_token.expand(B, -1, -1)
        template_tokens = template_tokens + F.layer_norm(
            template_embed.mean(dim=1, keepdim=True),
            template_tokens.shape[1:]
        )

        # Combine sequences and add position embeddings
        x = torch.cat([x, template_tokens], dim=1)
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Remove template token and normalize
        x = x[:, :-1]
        x = self.norm(x)

        # Reshape for decoder
        H = W = self.image_size // self.patch_size
        x = x.reshape(B, H, W, self.embed_dim)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Progressive upsampling and decoding
        x = self.decoder(x)

        return x


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CausalWeatherTransformer().to(device)

    # Create sample inputs
    x = torch.randn(1, 3, 256, 256).to(device)
    template = torch.randn(1, 3, 256, 256).to(device)

    # Forward pass
    output = model(x, template)
    print(f"Output shape: {output.shape}")  # Should be [1, 3, 256, 256]
    print(f"Using device: {device}")
