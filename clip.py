import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        # Maps token IDs to continuous embedding vectors
        self.token_embedding = nn.Embedding(n_vocab, n_embd)

        # Learnable positional embeddings for each token position
        # Shape: (Max_Seq_Len, Embed_Dim)
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # tokens: (Batch_Size, Seq_Len)

        # Convert token IDs to embeddings
        # (Batch_Size, Seq_Len) → (Batch_Size, Seq_Len, Embed_Dim)
        x = self.token_embedding(tokens)

        # Add positional information to token embeddings
        # Broadcasting applies position embeddings across the batch
        x += self.position_embedding
        
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        # Layer normalization applied before self-attention (Pre-Norm)
        self.layernorm_1 = nn.LayerNorm(n_embd)

        # Multi-head self-attention block
        self.attention = SelfAttention(n_head, n_embd)

        # Layer normalization applied before feedforward network
        self.layernorm_2 = nn.LayerNorm(n_embd)

        # Feedforward network (MLP) with expansion factor 4
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # x: Input token embeddings
        # Shape → (Batch_Size, Seq_Len, Embed_Dim)

        # Save input for residual connection
        residue = x
        
        ### SELF-ATTENTION BLOCK ###

        # Normalize input before attention
        x = self.layernorm_1(x)
        
        # Apply causal self-attention (no access to future tokens)
        x = self.attention(x, causal_mask=True)
        
        # Residual connection preserves original token information
        x += residue

        ### FEEDFORWARD (MLP) BLOCK ###

        # Save input for second residual connection
        residue = x

        # Normalize before feedforward network
        x = self.layernorm_2(x)
        
        # Expand embedding dimension
        x = self.linear_1(x)
        
        # Apply QuickGELU activation (used in CLIP for efficiency)
        x = x * torch.sigmoid(1.702 * x)
        
        # Project back to original embedding dimension
        x = self.linear_2(x)
        
        # Residual connection after feedforward network
        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

        # Token + position embedding layer
        self.embedding = CLIPEmbedding(
            n_vocab=49408,
            n_embd=768,
            n_token=77
        )

        # Stack of Transformer encoder layers
        self.layers = nn.ModuleList([
            CLIPLayer(n_head=12, n_embd=768) for _ in range(12)
        ])

        # Final layer normalization
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # Ensure tokens are of integer type for embedding lookup
        tokens = tokens.type(torch.long)
        
        # Convert tokens to contextual embeddings
        # (Batch_Size, Seq_Len) → (Batch_Size, Seq_Len, Embed_Dim)
        state = self.embedding(tokens)

        # Pass through Transformer encoder layers
        for layer in self.layers:
            state = layer(state)

        # Apply final normalization to encoder output
        output = self.layernorm(state)
        
        return output
