import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    """
    Fuses the encoded representations from the Geometric and Topological towers
    into a unified latent space using a cross-attention/concatenation mechanism.
    """
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__()
        
        # We concatenate the features from the two towers, then project back to d_model
        # Alternatively, cross-attention could be used here. For simplicity and robustness,
        # we start with a concatenated projection followed by self-attention fusion layers.
        self.projection = nn.Linear(d_model * 2, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, geom_out, topo_out, mask=None):
        """
        geom_out: (batch, seq_len, d_model)
        topo_out: (batch, seq_len, d_model)
        """
        # Concatenate along the feature dimension
        combined = torch.cat((geom_out, topo_out), dim=-1)
        
        # Project down to latent space dimension
        projected = self.projection(combined)
        
        # Deep fusion via Self-Attention layers
        fused_latent = self.fusion_layers(projected, src_key_padding_mask=mask)
        
        return fused_latent
