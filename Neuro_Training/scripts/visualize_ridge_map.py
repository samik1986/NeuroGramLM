import os
import time
import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_ridge_map(
    raw_path="F0046_multichannel_cmle_ch03.tif",
    ridge_path="intensity_pipeline_outputs/ridge_map_volume.tif",
    output_png="ridge_map_visualization.png",
    artifact_png="/home/banerjee/.gemini/antigravity-ide/brain/de2cec35-726c-4959-bf87-f755d0bf5afb/ridge_map_visualization.png"
):
    print(f"Loading raw volume: {raw_path}")
    t0 = time.time()
    raw_vol = tifffile.imread(raw_path)
    print(f"Raw volume loaded in {time.time()-t0:.2f}s.")

    print(f"Loading Ridge Map volume: {ridge_path}")
    t1 = time.time()
    ridge_vol = tifffile.imread(ridge_path)
    print(f"Ridge Map volume loaded in {time.time()-t1:.2f}s.")

    print("Computing XY Maximum Intensity Projections...")
    
    # Cap intensity for better contrast
    p99 = np.percentile(raw_vol[::2, ::4, ::4], 99.8)
    if p99 == 0: p99 = 1.0
        
    ridge_p99 = np.percentile(ridge_vol[::2, ::4, ::4], 99.9)
    if ridge_p99 == 0: ridge_p99 = 50.0
    
    # Compute XY MIPs
    raw_mip = np.max(raw_vol, axis=0)
    ridge_mip = np.max(ridge_vol, axis=0)
    
    raw_mip_capped = np.clip(raw_mip, 0, p99).astype(np.float32) / float(p99)
    ridge_mip_capped = np.clip(ridge_mip, 0, ridge_p99).astype(np.float32) / float(ridge_p99)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    
    # Create an RGB image
    # We want the Raw volume in dim grayscale, and the Ridge Map in glowing CYAN (Green+Blue)
    H, W = raw_mip.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    
    dim_raw = raw_mip_capped * 0.4
    
    rgb[..., 0] = dim_raw  # Red
    rgb[..., 1] = np.clip(dim_raw + ridge_mip_capped * 1.5, 0, 1)  # Green
    rgb[..., 2] = np.clip(dim_raw + ridge_mip_capped * 1.5, 0, 1)  # Blue
    
    ax.imshow(rgb, origin='upper')
    
    # Create a custom legend/title
    ax.set_title("Hessian Ridge Filter Visualization\nCyan: Structural Ridges (DoG Bandpass) | Dim Gray: Raw Intensity", color='white', fontsize=14, pad=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_png, facecolor='black', dpi=300)
    print(f"Saved local preview to {output_png}")
    
    if artifact_png:
        os.makedirs(os.path.dirname(artifact_png), exist_ok=True)
        import shutil
        shutil.copyfile(output_png, artifact_png)
        print(f"Copied visualization to {artifact_png}")

if __name__ == "__main__":
    visualize_ridge_map()
