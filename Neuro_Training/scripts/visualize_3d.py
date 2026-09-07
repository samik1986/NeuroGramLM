import os
import sys
import numpy as np
import time
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_swc_3d(swc_path, output_png, title, color='cyan', subsample=1):
    print(f"Loading {swc_path} for 3D visualization...")
    if not os.path.exists(swc_path):
        print(f"File not found: {swc_path}")
        return

    nodes = {}
    with open(swc_path, 'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                p = line.split()
                if len(p) >= 7:
                    nodes[int(p[0])] = (float(p[2]), float(p[3]), float(p[4]), int(p[6]))

    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Hide axes for cleaner look
    ax.axis('off')
    
    # Collect line segments
    segments = []
    count = 0
    for nid, (x, y, z, pid) in nodes.items():
        if pid in nodes:
            count += 1
            if count % subsample == 0:
                px, py, pz, _ = nodes[pid]
                segments.append(([x, px], [y, py], [z, pz]))
                
    print(f"Plotting {len(segments)} segments...")
    for xs, ys, zs in segments:
        ax.plot(xs, ys, zs, color=color, linewidth=0.5, alpha=0.8)

    ax.set_title(title, color='white', pad=20, fontsize=16)
    
    # Set a nice viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_png, facecolor='black', dpi=300)
    plt.close()
    print(f"Saved {output_png}")

def plot_soma_3d(raw_path, masked_path, output_png, subsample=16):
    print("Loading TIFFs for 3D Soma Mask visualization...")
    if not os.path.exists(raw_path) or not os.path.exists(masked_path):
        print("TIFF files not found.")
        return
        
    t0 = time.time()
    raw_vol = tifffile.imread(raw_path)
    masked_vol = tifffile.imread(masked_path)
    print(f"Volumes loaded in {time.time()-t0:.2f}s")
    
    # Somas are where raw > threshold and masked == 0
    # To save memory, we can just look at subsampled slices
    z_coords, y_coords, x_coords = [], [], []
    
    print(f"Extracting soma locations (subsample={subsample})...")
    D, H, W = raw_vol.shape
    for z in range(0, D, subsample//2):
        slice_r = raw_vol[z, ::subsample, ::subsample]
        slice_m = masked_vol[z, ::subsample, ::subsample]
        
        # Somas were eroded and zeroed out in masked_vol
        mask = (slice_m == 0) & (slice_r > 500)
        
        y_idx, x_idx = np.where(mask)
        z_coords.extend([z] * len(y_idx))
        y_coords.extend(y_idx * subsample)
        x_coords.extend(x_idx * subsample)
        
    print(f"Found {len(z_coords)} soma points for 3D scatter.")
    
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.axis('off')
    
    ax.scatter(x_coords, y_coords, z_coords, color='red', s=0.5, alpha=0.5)
    
    ax.set_title("3D Soma Masks (Multi-Scale 4-50um)", color='white', pad=20, fontsize=16)
    ax.view_init(elev=30, azim=45)
    
    # Maintain aspect ratio (Z is physically smaller but plotting array indices)
    # Scale Z to visually match resolution ratio 0.5 / 0.1102 ~ 4.5
    ax.set_box_aspect((W, H, D * 4.5))
    
    plt.tight_layout()
    plt.savefig(output_png, facecolor='black', dpi=300)
    plt.close()
    print(f"Saved {output_png}")


def main():
    out_dir = "intensity_pipeline_outputs"
    soma_png = "3d_somas.png"
    pre_png = "3d_pre_bio_seeds.png"
    post_png = "3d_post_bio_seeds.png"
    
    raw_tif = "F0046_multichannel_cmle_ch03.tif"
    masked_tif = os.path.join(out_dir, "masked_volume.tif")
    pre_swc = os.path.join(out_dir, "unsupervised_seeds_pre_bio.swc")
    post_swc = os.path.join(out_dir, "unsupervised_seeds_post_bio.swc")
    
    plot_soma_3d(raw_tif, masked_tif, soma_png)
    plot_swc_3d(pre_swc, pre_png, "Pre-Bio Geometric Seeds (1st Pass)", color='cyan', subsample=2)
    plot_swc_3d(post_swc, post_png, "Post-Bio Validated Seeds (Clean Neurons)", color='lime', subsample=2)

    artifact_dir = "/home/banerjee/.gemini/antigravity-ide/brain/de2cec35-726c-4959-bf87-f755d0bf5afb/"
    import shutil
    for png in [soma_png, pre_png, post_png]:
        if os.path.exists(png):
            shutil.copyfile(png, os.path.join(artifact_dir, png))

if __name__ == "__main__":
    main()
