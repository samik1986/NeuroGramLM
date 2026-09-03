"""
Author: samik1986
Date: 2026-09-03
"""
import torch
import torch.nn as nn
from .embeddings import RotaryPositionEmbedding

class BaseDecoder(nn.Module):
    """
    Base class for modality-specific Transformer decoders.
    Performs masked self-attention and cross-attention over encoded memory.
    """
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.rope = RotaryPositionEmbedding(d_model // n_heads)
        self.n_heads = n_heads

    def apply_rope(self, x):
        batch, seq_len, d_model = x.shape
        x_reshaped = x.view(batch, seq_len, self.n_heads, d_model // self.n_heads).transpose(1, 2)
        q_embed, k_embed = self.rope(x_reshaped, x_reshaped)
        return q_embed.transpose(1, 2).reshape(batch, seq_len, d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None):
        """
        tgt: (batch, tgt_seq_len, d_model) The shifted target tokens.
        memory: (batch, src_seq_len, d_model) The output from the encoder / fusion.
        tgt_mask: Casual mask to prevent looking ahead.
        """
        tgt = self.apply_rope(tgt)
        out = self.decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return out


class GeometricDecoder(BaseDecoder):
    """
    Autoregressively generates the geometric sequence to bridge gaps.
    """
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__(d_model, n_heads, n_layers, dropout)


class TopologicalDecoder(BaseDecoder):
    """
    Autoregressively generates the topological sequence hierarchy.
    """
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__(d_model, n_heads, n_layers, dropout)
