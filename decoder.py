import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Normalize features across groups of channels for stability
        self.groupnorm = nn.GroupNorm(32, channels)

        # Self-attention over spatial tokens (no masking, full context)
        # Single attention head is used, matching Stable Diffusion VAE
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        # x: (Batch_Size, Channels, Height, Width)

        # Save input for residual (skip) connection
        residue = x 

        # Normalize features before attention
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        
        # Flatten spatial dimensions so each pixel becomes a token
        # (B, C, H, W) → (B, C, H*W)
        x = x.view((n, c, h * w))
        
        # Transpose to attention format:
        # tokens = H*W, embedding dimension = Channels
        # (B, C, H*W) → (B, H*W, C)
        x = x.transpose(-1, -2)
        
        # Apply self-attention across all spatial locations
        # Each pixel can attend to every other pixel
        x = self.attention(x)
        
        # Restore original tensor layout
        # (B, H*W, C) → (B, C, H*W)
        x = x.transpose(-1, -2)
        
        # Reshape back to image-like feature map
        # (B, C, H*W) → (B, C, H, W)
        x = x.view((n, c, h, w))
        
        # Residual connection preserves local structure
        x += residue

        return x 


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First normalization + convolution
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Second normalization + convolution
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual projection:
        # If channel dimensions differ, use 1×1 conv to match shapes
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x: (Batch_Size, In_Channels, Height, Width)

        # Save input for residual path
        residue = x

        # Normalize and apply non-linearity
        x = self.groupnorm_1(x)
        x = F.silu(x)
        
        # First feature transformation
        x = self.conv_1(x)
        
        # Normalize and apply non-linearity again
        x = self.groupnorm_2(x)
        x = F.silu(x)
        
        # Second feature transformation
        x = self.conv_2(x)
        
        # Add residual (skip) connection
        # Ensures structure preservation and stable gradients
        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # Channel-wise projection (no spatial change)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # Expand latent channels into high-dimensional feature space
            # (B, 4, H/8, W/8) → (B, 512, H/8, W/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # Deep residual refinement at lowest resolution
            VAE_ResidualBlock(512, 512), 
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Upsample spatial resolution (×2)
            # (B, 512, H/8, W/8) → (B, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),
            
            # Post-upsampling convolution to refine features
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # Residual refinement at higher resolution
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Upsample again
            # (B, 512, H/4, W/4) → (B, 512, H/2, W/2)
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # Gradually reduce channel dimensionality
            VAE_ResidualBlock(512, 256), 
            VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 
            
            # Final upsampling to full image resolution
            # (B, 256, H/2, W/2) → (B, 256, H, W)
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            # Final residual refinement before RGB projection
            VAE_ResidualBlock(256, 128), 
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 
            
            # Normalize and activate before output
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            
            # Project features back into RGB image space
            # (B, 128, H, W) → (B, 3, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        # x: Latent tensor sampled by the encoder
        # Shape → (Batch_Size, 4, Height/8, Width/8)
        
        # Undo the latent scaling applied in the encoder
        x /= 0.18215

        # Sequentially apply all decoder layers
        for module in self:
            x = module(x)

        # Output reconstructed image
        # (Batch_Size, 3, Height, Width)
        return x
