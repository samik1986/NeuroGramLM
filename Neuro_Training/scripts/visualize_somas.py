import os
import time
import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_somas(
    raw_path="F0046_multichannel_cmle_ch03.tif",
    masked_path="intensity_pipeline_outputs/masked_volume.tif",
    output_png="soma_mask_visualization.png",
    artifact_png="/home/banerjee/.gemini/antigravity-ide/brain/de2cec35-726c-4959-bf87-f755d0bf5afb/soma_mask_visualization.png"
):
    print(f"Loading raw volume: {raw_path}")
    t0 = time.time()
    raw_vol = tifffile.imread(raw_path)
    print(f"Raw volume loaded in {time.time()-t0:.2f}s.")

    print(f"Loading masked volume: {masked_path}")
    t1 = time.time()
    masked_vol = tifffile.imread(masked_path)
    print(f"Masked volume loaded in {time.time()-t1:.2f}s.")

    print("Computing MIPs...")
    # Cap intensity for better contrast
    p99 = np.percentile(raw_vol[::2, ::4, ::4], 99.8)
    if p99 == 0: p99 = 1.0
    
    # Isolate somas: regions where masked_vol is zero but raw_vol was high
    # Actually, a simpler difference: diff = raw_vol - masked_vol
    # But since it's uint16, we do np.where or cast to float
    diff_vol = np.where(masked_vol < raw_vol, raw_vol, 0)
    
    # Compute XY MIPs
    raw_mip = np.max(raw_vol, axis=0)
    soma_mip = np.max(diff_vol, axis=0)
    
    raw_mip_capped = np.clip(raw_mip, 0, p99).astype(np.float32) / float(p99)
    soma_mip_capped = np.clip(soma_mip, 0, p99).astype(np.float32) / float(p99)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    
    # Create an RGB image
    # Red channel = somas
    # Green & Blue channels = raw_mip (gives it a cyan/white backdrop, or we can do raw in green, somas in red)
    # Let's make raw grayscale (all channels equal) and add red for somas.
    
    H, W = raw_mip.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(raw_mip_capped + soma_mip_capped * 1.5, 0, 1)  # Red gets extra boost from somas
    rgb[..., 1] = raw_mip_capped * 0.7  # Dimmer background
    rgb[..., 2] = raw_mip_capped * 0.7
    
    ax.imshow(rgb, origin='upper')
    
    # Create a custom legend/title
    ax.set_title("Soma Masking Visualization\nRed: Masked Somas (Removed) | Gray: Remaining Neuronal Signal", color='white', fontsize=14, pad=10)
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
    visualize_somas()
