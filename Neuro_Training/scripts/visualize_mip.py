import argparse
import os
import numpy as np
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_mip_overlay(tiff_path, swc_path, base_swc_path, output_png):
    print(f"Loading TIFF volume from: {tiff_path}")
    vol = tifffile.imread(tiff_path)
    
    # Calculate Z-MIP (Maximum Intensity Projection)
    print("Calculating Maximum Intensity Projection (MIP) along Z-axis...")
    mip = np.max(vol, axis=0)
    
    # Increase contrast of MIP (clip top 1% to make background brighter)
    p99 = np.percentile(mip, 99.5)
    mip = np.clip(mip, 0, p99)
    
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    # cmap='gray' or 'magma' works well. 'gray' is requested for raw background.
    ax.imshow(mip, cmap='gray', origin='upper')
    
    print(f"Loading Base SWC: {base_swc_path}")
    b_df = pd.read_csv(base_swc_path, sep=r'\s+', comment='#', header=None,
                       names=['id', 'type', 'x', 'y', 'z', 'r', 'pid'])
    b_dict = {row['id']: row for _, row in b_df.iterrows()}
    base_edges = set()
    for _, row in b_df.iterrows():
        pid = row['pid']
        if pid != -1:
            base_edges.add((int(row['id']), int(pid)))
            
    print(f"Loading Final SWC: {swc_path}")
    swc_df = pd.read_csv(swc_path, sep=r'\s+', comment='#', header=None,
                         names=['id', 'type', 'x', 'y', 'z', 'r', 'pid'])
    node_dict = {row['id']: row for _, row in swc_df.iterrows()}
    
    print("Plotting Skeletons on MIP...")
    # We will collect line segments to plot efficiently using LineCollection
    from matplotlib.collections import LineCollection
    
    base_lines = []
    generated_lines = []
    
    # Physical resolution scaling factors
    res_x = 0.1102
    res_y = 0.1102
    
    for _, row in swc_df.iterrows():
        pid = row['pid']
        child_id = int(row['id'])
        if pid != -1 and pid in node_dict:
            parent = node_dict[pid]
            # Convert physical coordinates back to voxel coordinates for the image overlay
            x0, y0 = parent['x'] / res_x, parent['y'] / res_y
            x1, y1 = row['x'] / res_x, row['y'] / res_y
            
            line = [(x0, y0), (x1, y1)]
            
            # Check if edge is in base
            if (child_id, int(pid)) in base_edges:
                base_lines.append(line)
            else:
                generated_lines.append(line)
                
    # Plot Base lines (Green)
    lc_base = LineCollection(base_lines, colors='lime', linewidths=1.5, alpha=0.8, label='Neurons from Detection Phase')
    ax.add_collection(lc_base)
    
    # Plot Generated lines (Yellow)
    lc_gen = LineCollection(generated_lines, colors='yellow', linewidths=2.0, alpha=1.0, label='LLM Generated Neurons')
    ax.add_collection(lc_gen)
    
    ax.set_title('Maximum Intensity Projection (MIP) with Final Bridged Skeletons', fontsize=16, color='white')
    ax.axis('off')
    
    # Add a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lime', lw=2, label='Neurons from Detection Phase'),
        Line2D([0], [0], color='yellow', lw=2, label='LLM Generated Neurons')
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='black', labelcolor='white')
    
    # Set background color to black
    fig.patch.set_facecolor('black')
    
    print(f"Saving MIP overlay to {output_png}...")
    plt.tight_layout()
    plt.savefig(output_png, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff', default='F0046_multichannel_cmle_ch03.tif')
    parser.add_argument('--swc', default='intensity_pipeline_outputs/final.swc')
    parser.add_argument('--base_swc', default='intensity_pipeline_outputs/unsupervised_seeds_post_bio_scaled.swc')
    parser.add_argument('--out', default='mip_overlay.png')
    args = parser.parse_args()
    
    plot_mip_overlay(args.tiff, args.swc, args.base_swc, args.out)
