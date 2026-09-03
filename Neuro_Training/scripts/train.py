"""
Author: samik1986
Date: 2026-09-03
"""
import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import argparse

# Append the parent directory to sys.path so we can import Neuro_Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Neuro_Model.model import NeuroGramLM

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    # Progress bar for the epoch
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        # Move batch and targets to device
        batch = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                 for k, v in batch['inputs'].items()}
        targets = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                   for k, v in batch['targets'].items()}
                   
        padding_mask = batch.get('padding_mask', None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)
            
        tgt_mask = batch.get('tgt_mask', None)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)
            
        optimizer.zero_grad()
        
        # Forward pass (Bio is bypassed internally during training)
        outputs = model(batch, targets=targets, padding_mask=padding_mask, tgt_mask=tgt_mask)
        
        loss = outputs['loss']
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="NeuroGramLM Training Script")
    parser.add_argument('--resume_from', type=str, default=None, help="Path to checkpoint (.pt) for incremental training")
    args = parser.parse_args()

    # Setup
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    model_config_path = os.path.join(os.path.dirname(__file__), '../../Neuro_Model/config.json')
    
    train_config = load_config(config_path)
    model_config = load_config(model_config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    model = NeuroGramLM(model_config)
    
    if args.resume_from is not None:
        if os.path.exists(args.resume_from):
            print(f"Loading checkpoint for incremental training: {args.resume_from}")
            model.load_state_dict(torch.load(args.resume_from, map_location=device))
        else:
            print(f"Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")
            
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=train_config['training_parameters']['learning_rate'],
        weight_decay=train_config['training_parameters']['weight_decay']
    )
    
    # Dataloader (Mocked for architecture scaffolding)
    # dataloader = DataLoader(NeuroDataset(train_config['io_paths']['token_input_dir']), ...)
    print("WARNING: Dataloader is a placeholder. Implement NeuroDataset.")
    dataloader = [] 
    
    # Training Loop
    epochs = train_config['training_parameters']['epochs']
    save_dir = train_config['io_paths']['model_checkpoint_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        if len(dataloader) > 0:
            avg_loss = train_epoch(model, dataloader, optimizer, device)
            print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} skipped (Empty Dataloader)")
            
        if epoch % train_config['training_parameters']['save_every'] == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))
            
    print("Training Complete.")

if __name__ == "__main__":
    main()
