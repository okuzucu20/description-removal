import torch

import torch.nn as nn
import torch.nn.functional as F
from datatypes import TransformerNetworkOutput

class TransformerNetwork(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int = 2, dropout: float = 0.1):
        super(TransformerNetwork, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_layers = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.foreground_proj = nn.Linear(embed_dim, embed_dim)
        self.background_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, foreground_embed: torch.Tensor, background_embed: torch.Tensor) -> torch.Tensor:
        # Combine the embeddings and process through transformer layers
        combined = torch.stack([foreground_embed, background_embed], dim=0)  # Shape: (2, 1, 1024)
        combined = self.transformer_layers(combined)
        
        # Separate into foreground and background after processing
        processed_foreground = combined[0, :, :]  # Shape: (1, 1024)
        processed_background = combined[1, :, :]  # Shape: (1, 1024)

        # Project the processed embeddings to get final foreground and background embeddings
        foreground_output = self.foreground_proj(processed_foreground)
        background_output = self.background_proj(processed_background)

        return TransformerNetworkOutput(foreground_output=foreground_output, background_output=background_output)

# Example usage
if __name__ == "__main__":
    embed_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 2

    model = TransformerNetwork(embed_dim, num_heads, num_layers)

    foreground_embed = torch.rand(1, embed_dim)  # Example foreground embedding
    background_embed = torch.rand(1, embed_dim)  # Example background embedding

    output = model(foreground_embed, background_embed)
    foreground_output, background_output = output.foreground_output, output.background_output

    print(f"Foreground output shape: {foreground_output.shape}")
    print(f"Background output shape: {background_output.shape}")
