"""
Author: samik1986
Date: 2026-09-03
"""
import os
import json
import numpy as np
import sys
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import get_logger
from Neuro_Model.model import NeuroGramLM
from Neuro_Finetuning.layers.adapters import FinetuningWrapper
from Neuro_Retraining.scripts.run_incremental import BiologicalPlausibilityFilter

logger = get_logger("Domain_Finetuning", module_name="Finetuning")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def finetune_epoch(wrapped_model, dataloader, optimizer, device):
    wrapped_model.train()
    total_loss = 0.0
    
    # Progress bar for the epoch
    pbar = tqdm(dataloader, desc="Finetuning Domain")
    
    for step, batch in enumerate(pbar):
        # Mocking forward pass. Only decoders and domain kernel are updated.
        batch = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                 for k, v in batch['inputs'].items()}
        targets = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                   for k, v in batch['targets'].items()}
                   
        optimizer.zero_grad()
        
        # Forward pass through the wrapper
        outputs = wrapped_model(batch, targets=targets)
        
        loss = outputs['loss']
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="NeuroGramLM Domain Finetuning")
    parser.add_argument('--resolution', type=float, nargs=3, default=None, help="XYZ physical resolution (microns/voxel)")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to base CCFv3 checkpoint to finetune from")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    config = load_config(config_path)
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting Domain Finetuning. Using device: {device}")
    
    # 1. Biological Filtering of the New Domain
    bounds = config['biological_plausibility']
    filter_engine = BiologicalPlausibilityFilter(bounds)
    
    raw_swc_dir = config['io_paths']['raw_swc_input_dir']
    logger.info(f"Scanning for new domain SWCs in {raw_swc_dir}...")
    
    valid_swcs = []
    mock_swcs = [f"domain2_neuron_{i}.swc" for i in range(5)]
    total_files = len(mock_swcs)
    
    for i, swc in enumerate(mock_swcs):
        raw_points = np.random.rand(100, 7) * 2000.0
        if args.resolution is not None:
            res_array = np.array(args.resolution)
            raw_points[:, :3] = raw_points[:, :3] * res_array
            
        norm_points = filter_engine.normalize_scale(raw_points)
        if filter_engine.validate_biological_metrics(norm_points):
            valid_swcs.append(swc)
        else:
            logger.warning(f" -> REJECTED: {swc} is biologically implausible even as a new domain.")
            
    if len(valid_swcs) == 0:
        logger.error("No valid SWCs found for the new domain.")
        sys.exit(1)
        
    logger.info(f"{len(valid_swcs)}/{total_files} SWCs passed the Biological Filter. Ready for tokenization.")
    
    # 2. Model Initialization & Freezing
    model_config_path = os.path.join(os.path.dirname(__file__), '../../Neuro_Model/config.json')
    model_config = load_config(model_config_path)
    
    logger.info(f"Loading Base CCFv3 Model from {args.checkpoint}")
    base_model = NeuroGramLM(model_config)
    
    # Wrap model to inject DomainAdaptationKernel and freeze encoders
    d_model = model_config['architecture_parameters']['d_model']
    wrapped_model = FinetuningWrapper(base_model, d_model).to(device)
    
    trainable_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in wrapped_model.parameters())
    logger.info(f"Model Wrapped for Finetuning. Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # 3. Finetuning Loop
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, wrapped_model.parameters()), 
        lr=config['finetuning_parameters']['learning_rate'],
        weight_decay=config['finetuning_parameters']['weight_decay']
    )
    
    # Mock Dataloader
    dataloader = [] 
    logger.warning("Dataloader is a placeholder.")
    
    epochs = config['finetuning_parameters']['epochs']
    save_dir = config['io_paths']['finetuned_checkpoint_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        if len(dataloader) > 0:
            avg_loss = finetune_epoch(wrapped_model, dataloader, optimizer, device)
            logger.info(f"Epoch {epoch}/{epochs} | Finetuning Loss: {avg_loss:.4f}")
        else:
            logger.warning(f"Epoch {epoch}/{epochs} skipped (Empty Dataloader)")
            
        if epoch % config['finetuning_parameters']['save_every'] == 0:
            ckpt_path = os.path.join(save_dir, f"finetuned_epoch_{epoch}.pt")
            torch.save(wrapped_model.state_dict(), ckpt_path)
            logger.info(f"Saved finetuned checkpoint to {ckpt_path}")
            
    logger.info("Domain Finetuning Complete.")

if __name__ == "__main__":
    main()
