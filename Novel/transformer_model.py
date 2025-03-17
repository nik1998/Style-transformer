import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        
        # More gradual downsampling with spatial feature preservation
        self.conv_layers = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(in_channels, embed_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            
            # First downsampling (1/2)
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            
            # Second downsampling (1/4)
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),

            # Final downsampling to patch size (1/patch_size)
            nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=4, padding=0),
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
        return x, (H, W)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 32, 32))
        
    def forward(self, query, key_value):
        """
        Cross-attention mechanism where query is the template and key_value are the image patches
        
        Args:
            query: Template embeddings [B, Q, C]
            key_value: Image patch embeddings [B, N, C]
        """
        B, N, C = key_value.shape
        _, Q, _ = query.shape
        
        # Project queries, keys, and values
        q = self.q_proj(query)  # [B, Q, C]
        k = self.k_proj(key_value)  # [B, N, C]
        v = self.v_proj(key_value)  # [B, N, C]
        
        # Reshape for multi-head attention
        q = q.view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Q, D]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Q, N]
        
        # Add relative position bias
        rel_pos = F.interpolate(
            self.rel_pos_bias.unsqueeze(0),
            size=(Q, N),
            mode='bilinear',
            align_corners=True
        )
        attn = attn + rel_pos
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, Q, D]
        
        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, Q, C)  # [B, Q, C]
        out = self.dropout(self.out_proj(out))
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1_query = nn.LayerNorm(embed_dim)
        self.norm1_kv = nn.LayerNorm(embed_dim)
        self.attn = CrossAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP with reduced complexity
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )
        
    def forward(self, query, key_value):
        """
        Args:
            query: Template embeddings [B, Q, C]
            key_value: Image patch embeddings [B, N, C]
        """
        # Cross-attention
        q_norm = self.norm1_query(query)
        kv_norm = self.norm1_kv(key_value)
        attn_output = self.attn(q_norm, kv_norm)
        
        # Residual connection for query
        query = query + attn_output
        
        # MLP
        query = query + self.mlp(self.norm2(query))
        
        return query


class ConvolutionalDecoder(nn.Module):
    def __init__(self, embed_dim=256, out_channels=3):
        super().__init__()
        
        # Convert channel information to spatial dimensions early
        self.expanding_path = nn.Sequential(
            # First expansion block (16x16 -> 64x64)
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim * 4),
            nn.GELU(),
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            
            # Second expansion block (64x64 -> 256x256)
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.GELU(),
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
        # Simple contracting path to output channels
        self.contracting_path = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Expanding path
        x = self.expanding_path(x)
        
        # Contracting path
        x = self.contracting_path(x)
        
        return x


class WeatherTransformer(nn.Module):
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
        
        # Position embeddings for image patches
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Initialize position embeddings with 2D sinusoidal encoding
        self._init_pos_embed()
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Output processing
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = ConvolutionalDecoder(embed_dim, in_channels)
        
        # Initialize remaining weights
        self._init_weights()
        
    def _init_pos_embed(self):
        H = W = self.image_size // self.patch_size
        grid_h = torch.arange(H, dtype=torch.float32)
        grid_w = torch.arange(W, dtype=torch.float32)
        
        grid_h = grid_h / (H - 1) * 2 - 1
        grid_w = grid_w / (W - 1) * 2 - 1
        
        pos_h = torch.meshgrid(grid_h, grid_w, indexing='ij')[0].reshape(-1)
        pos_w = torch.meshgrid(grid_h, grid_w, indexing='ij')[1].reshape(-1)
        
        # Create sinusoidal position embeddings
        dim = self.embed_dim // 4
        omega = torch.exp(
            torch.arange(dim, dtype=torch.float32) *
            (-math.log(10000.0) / dim)
        )
        
        pos_h = pos_h.unsqueeze(1) * omega.unsqueeze(0)  # [H*W, dim]
        pos_w = pos_w.unsqueeze(1) * omega.unsqueeze(0)  # [H*W, dim]
        
        pos_h = torch.cat([torch.sin(pos_h), torch.cos(pos_h)], dim=1)  # [H*W, 2*dim]
        pos_w = torch.cat([torch.sin(pos_w), torch.cos(pos_w)], dim=1)  # [H*W, 2*dim]
        
        pos_emb = torch.cat([pos_h, pos_w], dim=1)  # [H*W, 4*dim]
        
        # Project to full embedding dimension with correct shapes
        projection = torch.randn(4 * dim, self.embed_dim)  # [4*dim, embed_dim]
        pos_emb = torch.matmul(pos_emb, projection)  # [H*W, embed_dim]
        
        self.pos_embed.data[0, :] = pos_emb
        
    def _init_weights(self):
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
        
        # Encode input image
        x, (H, W) = self.encoder(img)
        
        # Add position embeddings to image patches
        x = x + self.pos_embed
        
        # Encode template as query
        query, (H_q, W_q) = self.template_encoder(template)
        
        # Apply transformer blocks with cross-attention
        for block in self.blocks:
            query = block(query, x)
            
        # Normalize query
        query = self.norm(query)
        
        # Reshape query for decoder (use the full query for spatial dimensions)
        query_reshaped = query.reshape(B, H_q, W_q, self.embed_dim)
        query_reshaped = query_reshaped.permute(0, 3, 1, 2)  # [B, C, H_q, W_q]
        
        # Progressive upsampling and decoding
        output = self.decoder(query_reshaped)
        
        # Add residual connection and normalize to [0,1] range
        output = output + img
        return torch.clamp(output, 0.0, 1.0)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherTransformer().to(device)
    
    # Create sample inputs
    x = torch.randn(1, 3, 256, 256).to(device)
    template = torch.randn(1, 3, 256, 256).to(device)
    
    # Forward pass
    output = model(x, template)
