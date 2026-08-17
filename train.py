"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python train.py
Input: SWC tokenized dataset
Output: Trained NeuroGram Transformer model weights
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NeuroGramDataset, SPECIAL_TOKENS
from transformer_model import NeuroGramTransformer
import time
from typing import Tuple

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    
    criterion_inv = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<PAD>"])
    criterion_geo = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<PAD>"])
    criterion_reg = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<PAD>"])
    
    batch_count = 0
    start = time.time()
    
    for tokens, pes in dataloader:
        tokens = tokens.to(device)
        pes = pes.to(device)
        
        src = tokens[:, :-1, :]
        targets = tokens[:, 1:, :]
        src_pe = pes[:, :-1, :]
        
        pad_mask = (src[:, :, 1] == SPECIAL_TOKENS["<PAD>"])
        
        seq_len = src.size(1)
        causal_mask = generate_square_subsequent_mask(seq_len).to(device)
        
        optimizer.zero_grad()
        
        inv_logits, geo_logits, reg_logits = model(src, src_pe, padding_mask=pad_mask, causal_mask=causal_mask)
        
        inv_loss = criterion_inv(inv_logits.reshape(-1, inv_logits.size(-1)), targets[:, :, 0].reshape(-1))
        geo_loss = criterion_geo(geo_logits.reshape(-1, geo_logits.size(-1)), targets[:, :, 1].reshape(-1))
        reg_loss = criterion_reg(reg_logits.reshape(-1, reg_logits.size(-1)), targets[:, :, 2].reshape(-1))
        
        loss = inv_loss + geo_loss + reg_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        if batch_count % 10 == 0:
            print(f"Batch {batch_count} | Loss: {loss.item():.4f} | Time: {time.time()-start:.2f}s")
            start = time.time()
            
    return total_loss / max(1, batch_count)

def main() -> None:
    print("Setting up training pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = NeuroGramTransformer(
        geo_vocab_size=512 + 8,
        inv_vocab_size=512 + 8,
        reg_vocab_size=65000 + 8,
        d_model=512,
        n_heads=8,
        num_layers=6
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataset = NeuroGramDataset("dummy_dataset.jsonl", max_length=128) # small context for fast test
    dataloader = DataLoader(dataset, batch_size=4)
    
    print("Running a dummy training epoch to check for OOM and gradient flow...")
    loss = train_epoch(model, dataloader, optimizer, device)
    print(f"Training Loop Test Success! Average Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
