import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Single linear layer that jointly computes Query, Key, and Value
        # Equivalent to having separate Wq, Wk, Wv matrices
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # Output projection Wo that mixes information across heads
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        # Dimensionality per attention head
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: Input sequence of embeddings
        # Shape → (Batch_Size, Seq_Len, Embed_Dim)

        # Preserve original shape for final reshaping
        input_shape = x.shape
        
        batch_size, sequence_length, d_embed = input_shape

        # Shape used to split embeddings into multiple attention heads
        # (Batch_Size, Seq_Len, Num_Heads, Head_Dim)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # Project input into Query, Key, and Value in one operation
        # (B, Seq_Len, Embed_Dim) → (B, Seq_Len, 3 * Embed_Dim)
        # Then split into Q, K, V each of shape (B, Seq_Len, Embed_Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # Reshape and transpose for multi-head attention
        # (B, Seq_Len, Embed_Dim) → (B, Num_Heads, Seq_Len, Head_Dim)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute scaled dot-product attention scores
        # (B, H, Seq_Len, Head_Dim) @ (B, H, Head_Dim, Seq_Len)
        # → (B, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            # Create upper-triangular mask to prevent attention to future tokens
            # Used in autoregressive settings
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        # Scale attention scores to stabilize gradients
        weight /= math.sqrt(self.d_head)

        # Normalize attention scores into probabilities
        weight = F.softmax(weight, dim=-1)

        # Weighted sum of values
        # (B, H, Seq_Len, Seq_Len) @ (B, H, Seq_Len, Head_Dim)
        # → (B, H, Seq_Len, Head_Dim)
        output = weight @ v

        # Recombine attention heads
        # (B, H, Seq_Len, Head_Dim) → (B, Seq_Len, H, Head_Dim)
        output = output.transpose(1, 2)

        # Merge head dimension back into embedding dimension
        # (B, Seq_Len, H, Head_Dim) → (B, Seq_Len, Embed_Dim)
        output = output.reshape(input_shape)

        # Final linear projection
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Query projection from latent (e.g., image features)
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)

        # Key and Value projections from context (e.g., text embeddings)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        # Output projection after attention
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        # Dimensionality per attention head
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x: Query sequence (latent features)
        # Shape → (Batch_Size, Seq_Len_Q, Dim_Q)
        #
        # y: Context sequence (e.g., text embeddings)
        # Shape → (Batch_Size, Seq_Len_KV, Dim_KV)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # Shape used for splitting into multiple heads
        # Seq_Len is inferred (-1) for Q and KV separately
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # Project latent features to Queries
        q = self.q_proj(x)

        # Project context features to Keys and Values
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Reshape and transpose for multi-head attention
        # Queries: (B, Seq_Len_Q, Embed_Dim) → (B, H, Seq_Len_Q, Head_Dim)
        q = q.view(interim_shape).transpose(1, 2)

        # Keys: (B, Seq_Len_KV, Embed_Dim) → (B, H, Seq_Len_KV, Head_Dim)
        k = k.view(interim_shape).transpose(1, 2)

        # Values: (B, Seq_Len_KV, Embed_Dim) → (B, H, Seq_Len_KV, Head_Dim)
        v = v.view(interim_shape).transpose(1, 2)
        
        # Compute attention scores between latent queries and context keys
        # (B, H, Seq_Len_Q, Head_Dim) @ (B, H, Head_Dim, Seq_Len_KV)
        # → (B, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # Scale scores for numerical stability
        weight /= math.sqrt(self.d_head)
        
        # Convert scores to attention probabilities
        weight = F.softmax(weight, dim=-1)
        
        # Weighted sum over context values
        # (B, H, Seq_Len_Q, Seq_Len_KV) @ (B, H, Seq_Len_KV, Head_Dim)
        # → (B, H, Seq_Len_Q, Head_Dim)
        output = weight @ v
        
        # Recombine attention heads
        # (B, H, Seq_Len_Q, Head_Dim) → (B, Seq_Len_Q, H, Head_Dim)
        output = output.transpose(1, 2).contiguous()
        
        # Merge head dimension back into embedding dimension
        # (B, Seq_Len_Q, H, Head_Dim) → (B, Seq_Len_Q, Embed_Dim)
        output = output.view(input_shape)
        
        # Final linear projection
        output = self.out_proj(output)

        return output
