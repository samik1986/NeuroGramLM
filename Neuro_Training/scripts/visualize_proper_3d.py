import os
import sys
import numpy as np
import time
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_proper_3d(
    raw_path="F0046_multichannel_cmle_ch03.tif",
    masked_path="intensity_pipeline_outputs/masked_volume.tif",
    ridge_path="intensity_pipeline_outputs/ridge_map_volume.tif",
    output_png="proper_3d_overlay.png",
    artifact_png="/home/banerjee/.gemini/antigravity-ide/brain/de2cec35-726c-4959-bf87-f755d0bf5afb/proper_3d_overlay.png",
    subsample_xy=32,
    subsample_z=4
):
    print("Loading TIFFs for Proper 3D Overlay...")
    if not all(os.path.exists(p) for p in [raw_path, masked_path, ridge_path]):
        print("Error: Missing TIFF volumes.")
        return
        
    t0 = time.time()
    raw_vol = tifffile.imread(raw_path)
    masked_vol = tifffile.imread(masked_path)
    ridge_vol = tifffile.imread(ridge_path)
    print(f"Volumes loaded in {time.time()-t0:.2f}s")
    
    # Downsample significantly to prevent RAM/matplotlib crashes
    # Z resolution is 0.5, XY is 0.11. So subsample Z much less than XY.
    D, H, W = raw_vol.shape
    raw_sub = raw_vol[::subsample_z, ::subsample_xy, ::subsample_xy]
    masked_sub = masked_vol[::subsample_z, ::subsample_xy, ::subsample_xy]
    ridge_sub = ridge_vol[::subsample_z, ::subsample_xy, ::subsample_xy]
    
    print("Extracting 3D Coordinates...")
    # 1. Raw volume background (faint signal)
    # Threshold slightly above noise floor
    p95 = np.percentile(raw_sub, 95)
    bg_mask = (raw_sub > p95)
    bg_z, bg_y, bg_x = np.where(bg_mask)
    
    # 2. Somas (Erased by masking, but present in raw)
    # We define somas where the raw volume was bright, but masked volume is now zero
    soma_mask = (masked_sub == 0) & (raw_sub > p95)
    sm_z, sm_y, sm_x = np.where(soma_mask)
    
    # 3. Ridge Map (Tubular paths)
    ridge_thresh = np.percentile(ridge_sub, 99.5)
    rm_z, rm_y, rm_x = np.where(ridge_sub > ridge_thresh)
    
    print(f"Plotting Background: {len(bg_z)} points, Somas: {len(sm_z)} points, Ridges: {len(rm_z)} points...")
    
    fig = plt.figure(figsize=(14, 14), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.axis('off')
    
    # Plot background (dim gray)
    ax.scatter(bg_x, bg_y, bg_z, color='gray', s=0.2, alpha=0.1, label='Raw Background')
    
    # Plot Somas (Red)
    ax.scatter(sm_x, sm_y, sm_z, color='red', s=4.0, alpha=0.5, label='Soma Masks')
    
    # Plot Ridges (Cyan)
    ax.scatter(rm_x, rm_y, rm_z, color='cyan', s=2.0, alpha=0.7, label='DoG Ridge Map')
    
    ax.set_title("3D Volumetric Point Cloud Overlay\nGray: Raw | Red: Somas | Cyan: Ridges", color='white', pad=20, fontsize=16)
    
    # Adjust aspect ratio based on physical geometry
    # Physical Z = D * 0.5, Physical XY = H * 0.11
    # We downsampled, so the plotted indices have a specific aspect ratio
    physical_d = D * 0.5
    physical_h = H * 0.1102
    physical_w = W * 0.1102
    
    ax.set_box_aspect((physical_w, physical_h, physical_d))
    ax.view_init(elev=20, azim=65)
    
    # Legend
    leg = ax.legend(facecolor='black', edgecolor='white', loc='upper right')
    for text in leg.get_texts(): text.set_color("white")
    
    plt.tight_layout()
    plt.savefig(output_png, facecolor='black', dpi=300)
    plt.close()
    print(f"Saved local preview to {output_png}")
    
    if artifact_png:
        os.makedirs(os.path.dirname(artifact_png), exist_ok=True)
        import shutil
        shutil.copyfile(output_png, artifact_png)
        print(f"Copied visualization to {artifact_png}")

if __name__ == "__main__":
    visualize_proper_3d()
