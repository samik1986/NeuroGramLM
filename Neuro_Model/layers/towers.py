"""
Author: samik1986
Date: 2026-09-03
"""
import torch
import torch.nn as nn
from .embeddings import RotaryPositionEmbedding

class BaseTower(nn.Module):
    """
    Base class for modality-specific Transformer encoders.
    """
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.rope = RotaryPositionEmbedding(d_model // n_heads)
        self.n_heads = n_heads

    def apply_rope(self, x):
        batch, seq_len, d_model = x.shape
        x_reshaped = x.view(batch, seq_len, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        q_embed, k_embed = self.rope(x_reshaped, x_reshaped)
        return q_embed.transpose(1, 2).reshape(batch, seq_len, d_model)

    def forward(self, x, mask=None):
        x = self.apply_rope(x)
        out = self.transformer(x, src_key_padding_mask=mask)
        return out


class GeometricTower(BaseTower):
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__(d_model, n_heads, n_layers, dropout)
        
    def forward(self, geometric_embeddings, mask=None):
        return super().forward(geometric_embeddings, mask)


class TopologicalTower(BaseTower):
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__(d_model, n_heads, n_layers, dropout)
        
    def forward(self, topological_embeddings, mask=None):
        return super().forward(topological_embeddings, mask)


class BioTower(BaseTower):
    """
    Encoder for processing biological intensity metadata from the 3D CNN.
    """
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__(d_model, n_heads, n_layers, dropout)
        
    def forward(self, bio_embeddings, mask=None):
        """
        Processes intensity ridge embeddings. Will often receive null/dropout masks during training.
        """
        return super().forward(bio_embeddings, mask)
