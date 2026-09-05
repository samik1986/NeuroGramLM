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
import glob
import random
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
# Append the parent directory to sys.path so we can import Neuro_Model and sibling scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.dirname(__file__))

try:
    from visualize import vq_to_synthetic_3d_coords, render_validation_picture, render_validation_gif
except ImportError:
    from .visualize import vq_to_synthetic_3d_coords, render_validation_picture, render_validation_gif

from Neuro_Model.model import NeuroGramLM
from Neuro_Dataloader.dataset import NeuroDataset
from Neuro_Dataloader.collate import neuro_collate_fn
from utils.logger import get_logger

logger = get_logger("Trainer", module_name="Training")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def train_epoch(model, dataloader, optimizer, scaler, device, save_steps=None, save_dir=None, epoch=None, writer=None):
    model.train()
    total_loss = 0.0
    
    # Progress bar for the epoch
    pbar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(pbar):
        # Extract masks before modifying inputs
        padding_mask = batch.get('padding_mask', None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)
            
        tgt_mask = batch.get('tgt_mask', None)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)

        # Move inputs and targets to device
        inputs = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                 for k, v in batch['inputs'].items()}
        targets = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                   for k, v in batch['targets'].items()}
                   
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with Automatic Mixed Precision (AMP)
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=device_type == 'cuda'):
            # Pass flat kwargs to avoid DataParallel nested dictionary scatter corruption
            flat_kwargs = {
                'vq_tortuosity': inputs['vq_ids']['tortuosity'],
                'vq_curvature': inputs['vq_ids']['curvature_energy'],
                'vq_inertia': inputs['vq_ids']['inertia_tensor'],
                'topo_strahler': inputs['topological_ids']['strahler_order'],
                'topo_wl': inputs['topological_ids']['wl_hash'],
                'bio_volumes': inputs.get('bio_volumes', None),
                'target_vq_tortuosity': targets['vq_ids']['tortuosity'],
                'target_vq_curvature': targets['vq_ids']['curvature_energy'],
                'target_vq_inertia': targets['vq_ids']['inertia_tensor'],
                'target_topo_strahler': targets['topological_ids']['strahler_order'],
                'target_topo_wl': targets['topological_ids']['wl_hash'],
                'target_vq_tortuosity_shifted': targets['vq_ids_shifted']['tortuosity'],
                'target_vq_curvature_shifted': targets['vq_ids_shifted']['curvature_energy'],
                'target_vq_inertia_shifted': targets['vq_ids_shifted']['inertia_tensor'],
                'target_topo_strahler_shifted': targets['topological_ids_shifted']['strahler_order'],
                'target_topo_wl_shifted': targets['topological_ids_shifted']['wl_hash'],
                'padding_mask': padding_mask
            }
            outputs = model(**flat_kwargs)
            
            # Since outputs might be scattered across GPUs and returned as a list/dict, 
            # we need to make sure we handle the loss correctly
            if isinstance(outputs, dict):
                loss = outputs['loss'].mean()
            else:
                loss = outputs.mean()
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Step-level TensorBoard logging
        if writer is not None:
            global_step = (epoch - 1) * len(dataloader) + step
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            if isinstance(outputs, dict):
                for k in ['geom_loss', 'topo_loss', 'fusion_loss', 'dec_loss']:
                    if k in outputs and outputs[k] is not None:
                        writer.add_scalar(f'SubLoss/{k}', outputs[k].mean().item(), global_step)
        
        # Save intermediate checkpoint
        if save_steps and save_dir and (step + 1) % save_steps == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_step_{step + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved intermediate checkpoint to {ckpt_path}")
            
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

def validate_epoch(model, dataloader, device, epoch=1, vis_dir=None):
    model.eval()
    total_loss = 0.0
    
    saved_visualizations = False
    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for step, batch in enumerate(pbar):
            padding_mask = batch.get('padding_mask', None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
                
            tgt_mask = batch.get('tgt_mask', None)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(device)

            inputs = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                     for k, v in batch['inputs'].items()}
            targets = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
                       for k, v in batch['targets'].items()}
                       
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type, enabled=device_type == 'cuda'):
                flat_kwargs = {
                    'vq_tortuosity': inputs['vq_ids']['tortuosity'],
                    'vq_curvature': inputs['vq_ids']['curvature_energy'],
                    'vq_inertia': inputs['vq_ids']['inertia_tensor'],
                    'topo_strahler': inputs['topological_ids']['strahler_order'],
                    'topo_wl': inputs['topological_ids']['wl_hash'],
                    'bio_volumes': inputs.get('bio_volumes', None),
                    'target_vq_tortuosity': targets['vq_ids']['tortuosity'],
                    'target_vq_curvature': targets['vq_ids']['curvature_energy'],
                    'target_vq_inertia': targets['vq_ids']['inertia_tensor'],
                    'target_topo_strahler': targets['topological_ids']['strahler_order'],
                    'target_topo_wl': targets['topological_ids']['wl_hash'],
                    'target_vq_tortuosity_shifted': targets['vq_ids_shifted']['tortuosity'],
                    'target_vq_curvature_shifted': targets['vq_ids_shifted']['curvature_energy'],
                    'target_vq_inertia_shifted': targets['vq_ids_shifted']['inertia_tensor'],
                    'target_topo_strahler_shifted': targets['topological_ids_shifted']['strahler_order'],
                    'target_topo_wl_shifted': targets['topological_ids_shifted']['wl_hash'],
                    'padding_mask': padding_mask,
                    'tgt_mask': tgt_mask
                }
                outputs = model(**flat_kwargs)
                
                if isinstance(outputs, dict):
                    loss = outputs['loss'].mean()
                else:
                    loss = outputs.mean()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Save visual outputs for the first validation batch of the epoch
            if vis_dir and not saved_visualizations and isinstance(outputs, dict) and 'geom_logits' in outputs:
                try:
                    epoch_vis_dir = os.path.join(vis_dir, f"epoch_{epoch:03d}")
                    os.makedirs(epoch_vis_dir, exist_ok=True)
                    
                    # Take up to 3 samples from the batch
                    batch_size_val = inputs['vq_ids']['tortuosity'].shape[0]
                    num_samples = min(3, batch_size_val)
                    
                    pred_tort = torch.argmax(outputs['geom_logits']['tortuosity'], dim=-1).cpu().numpy()
                    pred_curv = torch.argmax(outputs['geom_logits']['curvature_energy'], dim=-1).cpu().numpy()
                    pred_iner = torch.argmax(outputs['geom_logits']['inertia_tensor'], dim=-1).cpu().numpy()
                    
                    in_tort = targets['vq_ids_shifted']['tortuosity'].cpu().numpy()
                    in_curv = targets['vq_ids_shifted']['curvature_energy'].cpu().numpy()
                    in_iner = targets['vq_ids_shifted']['inertia_tensor'].cpu().numpy()
                    
                    pad_np = padding_mask.cpu().numpy() if padding_mask is not None else None
                    
                    for s_idx in range(num_samples):
                        # Filter out padding tokens
                        if pad_np is not None:
                            valid_len = np.sum(~pad_np[s_idx])
                            s_in_tort = in_tort[s_idx, :valid_len]
                            s_in_curv = in_curv[s_idx, :valid_len]
                            s_in_iner = in_iner[s_idx, :valid_len]
                            
                            s_out_tort = pred_tort[s_idx, :valid_len]
                            s_out_curv = pred_curv[s_idx, :valid_len]
                            s_out_iner = pred_iner[s_idx, :valid_len]
                        else:
                            s_in_tort = in_tort[s_idx]
                            s_in_curv = in_curv[s_idx]
                            s_in_iner = in_iner[s_idx]
                            
                            s_out_tort = pred_tort[s_idx]
                            s_out_curv = pred_curv[s_idx]
                            s_out_iner = pred_iner[s_idx]
                            
                        # Generate continuous 3D coordinate trajectories
                        in_coords = vq_to_synthetic_3d_coords(s_in_tort, s_in_curv, s_in_iner)
                        out_coords = vq_to_synthetic_3d_coords(s_out_tort, s_out_curv, s_out_iner)
                        
                        # 1. Save static comparison image (low-res)
                        pic_path = os.path.join(epoch_vis_dir, f"val_sample_{s_idx + 1}_comparison.png")
                        render_validation_picture(in_coords, out_coords, pic_path, epoch=epoch, sample_idx=s_idx + 1)
                        
                        # 2. Save step-by-step animated GIF (low-res)
                        gif_path = os.path.join(epoch_vis_dir, f"val_sample_{s_idx + 1}_growth.gif")
                        render_validation_gif(in_coords, out_coords, gif_path, epoch=epoch, max_frames=25, fps=6)
                        
                    logger.info(f"Saved validation comparison pictures & GIFs to {epoch_vis_dir}")
                    saved_visualizations = True
                except Exception as e:
                    logger.warning(f"Could not render validation visualizations: {e}")
            
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="NeuroGramLM Training Script")
    parser.add_argument('--resume_from', type=str, default=None, help="Path to checkpoint (.pt) for incremental training")
    parser.add_argument('--epochs', type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()

    # Setup
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    model_config_path = os.path.join(os.path.dirname(__file__), '../../Neuro_Model/config.json')
    
    train_config = load_config(config_path)
    model_config = load_config(model_config_path)
    
    if args.epochs is not None:
        train_config['training_parameters']['epochs'] = args.epochs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting Training Process. Using device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark for optimal performance.")
    
    # Initialize Model
    model = NeuroGramLM(model_config)
    
    if args.resume_from is not None:
        if os.path.exists(args.resume_from):
            logger.info(f"Loading checkpoint for incremental training: {args.resume_from}")
            model.load_state_dict(torch.load(args.resume_from, map_location=device))
        else:
            logger.warning(f"Checkpoint {args.resume_from} not found. Starting from scratch.")
            
    model = model.to(device)
    # For now, forcing single GPU as per user request to avoid DataParallel Hopper bugs
    if False and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel!")
        model = torch.nn.DataParallel(model)
    
    # Optimizer and Scaler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=train_config['training_parameters']['learning_rate'],
        weight_decay=train_config['training_parameters']['weight_decay']
    )
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    # Setup Dataloader
    token_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', train_config['io_paths']['token_input_dir']))
    all_files = glob.glob(os.path.join(token_dir, "*.json"))
    
    # Training Loop config
    epochs = train_config['training_parameters']['epochs']
    save_dir = train_config['io_paths']['model_checkpoint_dir']
    save_steps = train_config['training_parameters'].get('save_steps', 1000)
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard setup
    log_dir = train_config['io_paths'].get('log_dir', os.path.join(save_dir, '../logs/tensorboard'))
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logging to: {log_dir}")
    
    # Validation visualization setup
    vis_dir = train_config['io_paths'].get('val_vis_dir', os.path.join(save_dir, '../analysis/val_visualizations'))
    os.makedirs(vis_dir, exist_ok=True)
    logger.info(f"Validation visualizations will be saved to: {vis_dir}")

    for epoch in range(1, epochs + 1):
        # Randomize validation set each epoch
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * 0.8)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        train_dataset = NeuroDataset(token_dir, file_list=train_files)
        val_dataset = NeuroDataset(token_dir, file_list=val_files)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=train_config['training_parameters']['batch_size'], 
            shuffle=True, 
            collate_fn=neuro_collate_fn,
            num_workers=8,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=train_config['training_parameters']['batch_size'], 
            shuffle=False, 
            collate_fn=neuro_collate_fn,
            num_workers=8,
            pin_memory=True
        )
        
        if len(train_dataloader) > 0:
            avg_loss = train_epoch(model, train_dataloader, optimizer, scaler, device, save_steps=save_steps, save_dir=save_dir, epoch=epoch, writer=writer)
            logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f}")
            writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        else:
            logger.warning(f"Epoch {epoch}/{epochs} skipped (Empty Dataloader)")
            
        if epoch % 1 == 0:
            if len(val_dataloader) > 0:
                val_loss = validate_epoch(model, val_dataloader, device, epoch=epoch, vis_dir=vis_dir)
                logger.info(f"Epoch {epoch}/{epochs} | Validation Loss: {val_loss:.4f}")
                writer.add_scalar('Loss/validation', val_loss, epoch)
            else:
                logger.warning(f"Epoch {epoch}/{epochs} Validation skipped (Empty Dataloader)")
                
        if epoch % train_config['training_parameters']['save_every'] == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")
            
    writer.close()
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
