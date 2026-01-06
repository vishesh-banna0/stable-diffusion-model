import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Expand timestep embedding into higher-dimensional representation
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: Sinusoidal timestep embedding
        # Shape → (1, 320)

        # Project timestep to higher dimension
        # (1, 320) → (1, 1280)
        x = self.linear_1(x)
        
        # Non-linearity for richer conditioning
        x = F.silu(x)
        
        # Final time embedding used throughout the UNet
        # (1, 1280) → (1, 1280)
        x = self.linear_2(x)

        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        # Normalize and process spatial features
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Project timestep embedding to match feature channels
        self.linear_time = nn.Linear(n_time, out_channels)

        # Normalize and process merged feature + time information
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual projection if channel dimensions differ
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: Spatial feature map
        # Shape → (Batch_Size, In_Channels, Height, Width)
        #
        # time: Time embedding
        # Shape → (1, 1280)

        residue = feature
        
        # Normalize and activate spatial features
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        
        # Project features to output channel dimension
        feature = self.conv_feature(feature)
        
        # Process timestep embedding
        time = F.silu(time)
        time = self.linear_time(time)
        
        # Inject time information by broadcasting over spatial dimensions
        # (B, C, H, W) + (1, C, 1, 1)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # Normalize and activate merged representation
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        
        # Final convolution after time conditioning
        merged = self.conv_merged(merged)
        
        # Residual connection preserves spatial structure
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        # Normalize spatial features before attention
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)

        # Channel-wise projection before attention
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # Transformer-style attention stack
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(channels)

        # Feedforward network with GeGLU activation
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        # Project back to convolutional feature space
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: Spatial feature map
        # Shape → (Batch_Size, Channels, Height, Width)
        #
        # context: Text/context embeddings
        # Shape → (Batch_Size, Seq_Len, Dim)

        residue_long = x

        # Normalize and prepare for attention
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # Flatten spatial dimensions into a token sequence
        # (B, C, H, W) → (B, H*W, C)
        x = x.view((n, c, h * w)).transpose(-1, -2)
        
        # --- Self-Attention ---
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        
        # --- Cross-Attention (text conditioning) ---
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short
        
        # --- Feedforward with GeGLU ---
        residue_short = x
        x = self.layernorm_3(x)
        
        # Split for GeGLU gating
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        
        # Restore spatial layout
        # (B, H*W, C) → (B, C, H, W)
        x = x.transpose(-1, -2).view((n, c, h, w))

        # Final residual connection over the entire block
        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Convolution after spatial upsampling
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Nearest-neighbor upsampling followed by convolution
        # (B, C, H, W) → (B, C, 2H, 2W)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        # Dispatch inputs based on layer type
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder path (downsampling + feature extraction)
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        # Bottleneck at lowest spatial resolution
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )
        
        # Decoder path (upsampling + skip connections)
        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: Noisy latent
        # context: Text embeddings
        # time: Time embedding

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        # Bottleneck processing
        x = self.bottleneck(x, context, time)

        # Decoder with skip connections
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Normalize and activate final UNet features
        x = self.groupnorm(x)
        x = F.silu(x)
        
        # Project to noise prediction channels
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        # latent: Noisy latent tensor
        # context: Text embeddings
        # time: Timestep embedding

        # Expand timestep embedding
        time = self.time_embedding(time)
        
        # Predict noise using conditional UNet
        output = self.unet(latent, context, time)
        
        # Final projection to latent noise space
        output = self.final(output)
        
        return output
