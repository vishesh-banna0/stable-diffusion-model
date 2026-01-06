import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # Initial projection from RGB image space into feature space
            # (Batch_Size, 3, H, W) → (Batch_Size, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # Residual processing at full resolution
            # Preserves spatial size while enriching representations
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # First spatial downsampling (×2)
            # Padding handled manually in forward() for asymmetric behavior
            # (Batch_Size, 128, H, W) → (Batch_Size, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # Increase channel capacity after downsampling
            # (Batch_Size, 128, H/2, W/2) → (Batch_Size, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # Second spatial downsampling (×4 total)
            # (Batch_Size, 256, H/2, W/2) → (Batch_Size, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # Further increase in representational capacity
            # (Batch_Size, 256, H/4, W/4) → (Batch_Size, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            # Third spatial downsampling (×8 total)
            # This matches Stable Diffusion latent resolution
            # (Batch_Size, 512, H/4, W/4) → (Batch_Size, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # Deep residual processing in latent feature space
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # Self-attention over spatial latent tokens
            # Allows global context aggregation within the latent map
            VAE_AttentionBlock(512),

            # Final refinement before parameter projection
            VAE_ResidualBlock(512, 512),

            # Normalize channels for numerical stability
            nn.GroupNorm(32, 512),

            # Non-linear activation (smooth, stable for VAEs)
            nn.SiLU(),

            # Project features into latent distribution parameters
            # padding=1 preserves spatial size with kernel_size=3
            # Output has 8 channels: 4 for mean, 4 for log-variance
            # (Batch_Size, 512, H/8, W/8) → (Batch_Size, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # Channel-wise linear projection (no spatial effect)
            # Keeps parameter dimensionality intact
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        # x: Input image tensor
        #     Shape → (Batch_Size, 3, Height, Width)
        
        # noise: Standard Gaussian noise used for reparameterization
        #        Shape → (Batch_Size, 4, Height/8, Width/8)

        for module in self:

            # For strided convolutions, Stable Diffusion uses asymmetric padding
            # Padding is applied only on the right and bottom edges
            # This ensures exact spatial alignment with the decoder
            if getattr(module, 'stride', None) == (2, 2):
                # Pad format: (left, right, top, bottom)
                # (Batch_Size, C, H, W) → (Batch_Size, C, H+1, W+1)
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)

        # Split latent parameters into mean and log-variance
        # (Batch_Size, 8, H/8, W/8) → two tensors of
        # (Batch_Size, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamp log-variance to avoid numerical instability
        # Prevents extremely small or large variances during training/inference
        log_variance = torch.clamp(log_variance, -30, 20)

        # Convert log-variance to variance
        variance = log_variance.exp()

        # Standard deviation needed for reparameterization
        stdev = variance.sqrt()

        # Reparameterization trick:
        # Samples from N(mean, variance) using noise ~ N(0, 1)
        x = mean + stdev * noise

        # Latent scaling factor used by Stable Diffusion
        # Ensures latent magnitude matches training distribution
        # Source: CompVis Stable Diffusion config
        x *= 0.18215

        return x
