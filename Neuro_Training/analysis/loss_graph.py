"""
Author: samik1986
Date: 2026-09-03
"""
import json
import matplotlib.pyplot as plt
import os

def parse_logs(log_path):
    """
    Parses the JSONL training log file to extract epoch losses.
    """
    epochs = []
    geom_losses = []
    topo_losses = []
    fusion_losses = []
    dec_losses = []
    total_losses = []

    with open(log_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data['epoch'])
            geom_losses.append(data['loss_dict']['geometric'])
            topo_losses.append(data['loss_dict']['topological'])
            fusion_losses.append(data['loss_dict']['fusion'])
            
            # Decoder loss may not be present in early steps if testing encoders only
            dec_losses.append(data['loss_dict'].get('decoder', 0.0)) 
            total_losses.append(data['total_loss'])

    return epochs, geom_losses, topo_losses, fusion_losses, dec_losses, total_losses

def plot_loss_curves(log_path, save_dir):
    """
    Generates and saves the loss trajectory curves for all modality towers.
    """
    if not os.path.exists(log_path):
        print(f"Log file not found at {log_path}. Ensure training has run.")
        return

    epochs, geom, topo, fusion, dec, total = parse_logs(log_path)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, geom, label='Geometric Loss (Auxiliary)', alpha=0.7)
    plt.plot(epochs, topo, label='Topological Loss (Auxiliary)', alpha=0.7)
    plt.plot(epochs, fusion, label='Fusion Loss', alpha=0.7, linestyle='--')
    plt.plot(epochs, dec, label='Decoder Loss (Autoregressive)', linewidth=2)
    plt.plot(epochs, total, label='Total Combined Loss', color='black', linewidth=2.5)

    plt.title('NeuroGramLM Multimodal Training Loss Trajectories')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(out_path)
    print(f"Saved loss graph to: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Training Losses")
    parser.add_argument('--log', type=str, default='./logs/training_log.jsonl', help='Path to log file')
    parser.add_argument('--out', type=str, default='.', help='Directory to save the plot')
    args = parser.parse_args()
    
    plot_loss_curves(args.log, args.out)
