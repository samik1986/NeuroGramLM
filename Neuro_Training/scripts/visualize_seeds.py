import os
import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_seeds(tiff_path, swc_path, output_png):
    print(f"Loading TIFF volume: {tiff_path}")
    vol = tifffile.imread(tiff_path)
    D, H, W = vol.shape
    
    print("Computing XY Maximum Intensity Projection...")
    # Cap intensity for better contrast
    p99 = np.percentile(vol[::4, ::4, ::4], 99.9)
    if p99 == 0: p99 = 1.0
    vol_capped = np.clip(vol, 0, p99) / p99
    mip_xy = np.max(vol_capped, axis=0)
    
    print(f"Loading SWC seeds: {swc_path}")
    nodes = {}
    edges = []
    with open(swc_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 7:
                nid = int(parts[0])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                pid = int(parts[6])
                nodes[nid] = (x, y, z)
                if pid != -1:
                    edges.append((pid, nid))
    
    print(f"Loaded {len(nodes)} nodes, {len(edges)} connections.")
    
    # Check physical scaling
    max_c = np.max([v for v in nodes.values()], axis=0) if nodes else np.array([0,0,0])
    rx, ry, rz = 0.1102, 0.1102, 0.5
    scale_factor = 1.0
    if max_c.max() < max(W, H) * rx * 1.5:
        scale_factor = 1.0 / rx
        
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.imshow(mip_xy, cmap='magma', origin='upper')
    
    from matplotlib.collections import LineCollection
    lines = []
    for pid, cid in edges:
        if pid in nodes and cid in nodes:
            px, py = nodes[pid][0] * scale_factor, nodes[pid][1] * scale_factor
            cx, cy = nodes[cid][0] * scale_factor, nodes[cid][1] * scale_factor
            lines.append([(px, py), (cx, cy)])
            
    if lines:
        if len(lines) > 200000:
            import random
            lines = random.sample(lines, 200000)
            print("Plotting 200000 downsampled lines.")
        lc = LineCollection(lines, colors='#00FFFF', linewidths=0.8, alpha=0.9)
        ax.add_collection(lc)
    else:
        # Fallback to scatter if no edges
        pts = np.array([v for v in nodes.values()]) * scale_factor
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c='#00FFFF', s=2.5, alpha=0.9, edgecolors='none')
        
    ax.set_title(f"Unsupervised Fragment Seeds Overlay (MIP)\nTotal Seeds: {len(nodes)}", color='white', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_png, facecolor='black', dpi=300)
    print(f"Saved to {output_png}")

if __name__ == "__main__":
    visualize_seeds(
        "F0046_multichannel_cmle_ch03.tif",
        "intensity_pipeline_outputs/unsupervised_seeds.swc",
        "unsupervised_seeds_visualization.png"
    )
