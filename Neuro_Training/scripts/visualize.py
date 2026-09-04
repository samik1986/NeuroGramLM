"""
Author: samik1986
Date: 2026-09-04
Visualization utility to render 3D/2D neuron sequence trajectories and animated GIFs 
comparing input sequence fragments vs model output sequence fragments.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

def vq_to_synthetic_3d_coords(tortuosity_ids, curvature_ids, inertia_ids, step_length=2.0):
    """
    Reconstructs continuous 3D coordinate trajectories from discrete VQ token sequences.
    Simulates physical stepping vectors modulated by intrinsic tortuosity and curvature angles.
    """
    N = len(tortuosity_ids)
    if N == 0:
        return np.zeros((0, 3))
        
    coords = np.zeros((N, 3), dtype=np.float32)
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Initial forward direction unit vector
    heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    heading /= np.linalg.norm(heading)
    
    coords[0] = pos
    
    for i in range(1, N):
        # Normalize token IDs to pseudo physical angles
        tort = float(tortuosity_ids[i]) / 255.0  # [0, 1]
        curv = float(curvature_ids[i]) / 255.0   # [0, 1]
        iner = float(inertia_ids[i]) / 511.0     # [0, 1]
        
        # Perturb heading based on curvature and tortuosity
        pitch = (curv - 0.5) * np.pi * 0.5
        yaw = (tort - 0.5) * np.pi * 0.8
        roll = (iner - 0.5) * np.pi * 0.5
        
        # Simple directional update
        dx = np.cos(pitch) * np.cos(yaw)
        dy = np.sin(pitch) * np.cos(yaw)
        dz = np.sin(yaw) + 0.3 * np.sin(roll)
        
        step_vec = np.array([dx, dy, dz], dtype=np.float32)
        norm = np.linalg.norm(step_vec)
        if norm > 1e-6:
            step_vec /= norm
            
        pos = pos + step_vec * step_length * (1.0 + 0.5 * tort)
        coords[i] = pos
        
    return coords

def render_validation_picture(input_coords, output_coords, save_path, epoch, sample_idx=0):
    """
    Saves a low-resolution static comparison picture showing input fragments in Cyan/Blue
    and predicted output fragments in Coral/Orange.
    """
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Input Fragment
    if len(input_coords) > 0:
        ax.plot(
            input_coords[:, 0], input_coords[:, 1], input_coords[:, 2],
            color='#00D4FF', linewidth=2.0, alpha=0.85, label='Input Fragment (Target Shifted)'
        )
        ax.scatter(
            input_coords[0, 0], input_coords[0, 1], input_coords[0, 2],
            color='#0055FF', s=30, marker='o', label='Input Start'
        )

    # Plot Output Fragment
    if len(output_coords) > 0:
        ax.plot(
            output_coords[:, 0], output_coords[:, 1], output_coords[:, 2],
            color='#FF4500', linewidth=2.0, linestyle='--', alpha=0.9, label='Model Output Fragment'
        )
        ax.scatter(
            output_coords[-1, 0], output_coords[-1, 1], output_coords[-1, 2],
            color='#FFD700', s=30, marker='^', label='Output End'
        )
        
    ax.set_title(f"NeuroGramLM Validation Epoch {epoch} - Sample {sample_idx}", fontsize=11, fontweight='bold')
    ax.set_xlabel('X (a.u.)', fontsize=8)
    ax.set_ylabel('Y (a.u.)', fontsize=8)
    ax.set_zlabel('Z (a.u.)', fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    # Clean viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

def render_validation_gif(input_coords, output_coords, save_gif_path, epoch, max_frames=30, fps=8):
    """
    Renders an animated low-resolution GIF showing the sequential growth step of the output sequence.
    """
    total_len = max(len(input_coords), len(output_coords))
    if total_len == 0:
        return
        
    step_stride = max(1, total_len // max_frames)
    indices = list(range(2, total_len + 1, step_stride))
    if indices[-1] != total_len:
        indices.append(total_len)
        
    # Global bounding box for fixed viewport
    all_pts = np.vstack([c for c in [input_coords, output_coords] if len(c) > 0])
    min_b = all_pts.min(axis=0) - 2.0
    max_b = all_pts.max(axis=0) + 2.0
    
    frames = []
    
    for idx in indices:
        fig = plt.figure(figsize=(6, 4.5), dpi=80)
        ax = fig.add_subplot(111, projection='3d')
        
        # Static or evolving input
        in_slice = input_coords[:min(idx, len(input_coords))]
        if len(in_slice) > 0:
            ax.plot(in_slice[:, 0], in_slice[:, 1], in_slice[:, 2], color='#00D4FF', linewidth=2.0, label='Input Fragment')
            
        # Evolving output
        out_slice = output_coords[:min(idx, len(output_coords))]
        if len(out_slice) > 0:
            ax.plot(out_slice[:, 0], out_slice[:, 1], out_slice[:, 2], color='#FF4500', linewidth=2.0, linestyle='--', label='Model Output Fragment')
            ax.scatter(out_slice[-1, 0], out_slice[-1, 1], out_slice[-1, 2], color='#FFD700', s=25, marker='o')
            
        ax.set_xlim(min_b[0], max_b[0])
        ax.set_ylim(min_b[1], max_b[1])
        ax.set_zlim(min_b[2], max_b[2])
        
        ax.set_title(f"Validation Epoch {epoch} | Step {idx}/{total_len}", fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        ax.view_init(elev=20, azim=30 + (idx * 2) % 360)
        
        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        rgb_image = rgba_buffer[:, :, :3]
        frames.append(rgb_image)
        plt.close(fig)
        
    os.makedirs(os.path.dirname(save_gif_path), exist_ok=True)
    imageio.mimsave(save_gif_path, frames, fps=fps)
