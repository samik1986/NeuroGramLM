import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os

def plot_soma_masked_mip(tiff_path, output_png):
    print(f"Loading masked TIFF volume from: {tiff_path}")
    if not os.path.exists(tiff_path):
        print(f"Error: {tiff_path} not found.")
        return
        
    vol = tifffile.imread(tiff_path)
    
    print("Calculating Maximum Intensity Projection (MIP) along Z-axis...")
    mip = np.max(vol, axis=0)
    
    # Clip top percentile for visibility
    p99 = np.percentile(mip, 99.5)
    mip = np.clip(mip, 0, p99)
    
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    ax.imshow(mip, cmap='gray', origin='upper')
    
    ax.set_title('MIP of Soma-Masked Volume (Thick Dendrites Preserved)', fontsize=16, color='white')
    ax.axis('off')
    
    fig.patch.set_facecolor('black')
    
    print(f"Saving MIP to {output_png}...")
    plt.tight_layout()
    plt.savefig(output_png, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff', default='intensity_pipeline_outputs/masked_volume.tif')
    parser.add_argument('--out', default='masked_volume_mip.png')
    args = parser.parse_args()
    
    plot_soma_masked_mip(args.tiff, args.out)
