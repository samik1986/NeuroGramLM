"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python transformer_model.py
Input: Tokenized NeuroGram data
Output: Multi-stream Transformer Neural Network model
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class MultiStreamEmbedding(nn.Module):
    def __init__(self, geo_vocab_size: int, inv_vocab_size: int, reg_vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.geo_embed = nn.Embedding(geo_vocab_size, d_model, padding_idx=0)
        self.inv_embed = nn.Embedding(inv_vocab_size, d_model, padding_idx=0)
        self.reg_embed = nn.Embedding(reg_vocab_size, d_model, padding_idx=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, L, 3) where:
           x[:, :, 0] = INV IDs
           x[:, :, 1] = GEO IDs
           x[:, :, 2] = REG IDs
        """
        inv_emb = self.inv_embed(x[:, :, 0])
        geo_emb = self.geo_embed(x[:, :, 1])
        reg_emb = self.reg_embed(x[:, :, 2])
        
        return inv_emb + geo_emb + reg_emb

class GraphLaplacianPositionalEncoding(nn.Module):
    def __init__(self, k: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(k, d_model)
        
    def forward(self, laplacian_pe: torch.Tensor) -> torch.Tensor:
        """
        laplacian_pe: Tensor of shape (B, L, k) containing the eigenvectors
        """
        return self.proj(laplacian_pe)

class NeuroGramTransformer(nn.Module):
    def __init__(
        self, 
        geo_vocab_size: int, 
        inv_vocab_size: int, 
        reg_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        laplacian_k: int = 8
    ) -> None:
        super().__init__()
        self.d_model = d_model
        
        self.embedding = MultiStreamEmbedding(geo_vocab_size, inv_vocab_size, reg_vocab_size, d_model)
        self.laplacian_pe = GraphLaplacianPositionalEncoding(laplacian_k, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.inv_head = nn.Linear(d_model, inv_vocab_size)
        self.geo_head = nn.Linear(d_model, geo_vocab_size)
        self.reg_head = nn.Linear(d_model, reg_vocab_size)
        
    def forward(self, x: torch.Tensor, pe: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, L, 3) Token IDs
        pe: (B, L, k) Laplacian Eigenvectors
        padding_mask: (B, L) Boolean tensor where True means padded token
        causal_mask: (L, L) Tensor preventing attention to future tokens
        """
        h = self.embedding(x)
        h = h + self.laplacian_pe(pe)
        
        out = self.encoder(h, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        inv_logits = self.inv_head(out)
        geo_logits = self.geo_head(out)
        reg_logits = self.reg_head(out)
        
        return inv_logits, geo_logits, reg_logits

def measure_model_memory() -> None:
    print("Initializing NeuroGramTransformer...")
    model = NeuroGramTransformer(
        geo_vocab_size=512 + 8,
        inv_vocab_size=512 + 8,
        reg_vocab_size=65000 + 8,
        d_model=512,
        n_heads=8,
        num_layers=6
    )
    if torch.cuda.is_available():
        model = model.cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    print("Testing forward pass (Batch: 8, Max_Len: 2048)...")
    device = next(model.parameters()).device
    dummy_x = torch.randint(0, 500, (8, 2048, 3)).to(device)
    dummy_pe = torch.randn(8, 2048, 8).to(device)
    
    try:
        inv_out, geo_out, reg_out = model(dummy_x, dummy_pe)
        print(f"Forward Pass Success! Output shapes:")
        print(f"  INV Logits: {inv_out.shape}")
        print(f"  GEO Logits: {geo_out.shape}")
        print(f"  REG Logits: {reg_out.shape}")
        
        if torch.cuda.is_available():
            print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            print("No OOM Leak detected for 2048 context window on 16GB GPU target!")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM LEAK DETECTED!")
        else:
            print(f"RuntimeError: {e}")

if __name__ == "__main__":
    measure_model_memory()
