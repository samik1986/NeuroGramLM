import os
import sys
import time
import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def render_visualization(
    tiff_path="F0046_multichannel_cmle_ch03.tif",
    swc_path="intensity_pipeline_outputs/final_smoothed_neurons.swc",
    output_png="final_smoothed_neurons_overlay.png",
    artifact_png="/home/banerjee/.gemini/antigravity-ide/brain/de2cec35-726c-4959-bf87-f755d0bf5afb/final_smoothed_neurons_overlay.png"
):
    print(f"Loading TIFF volume: {tiff_path}")
    t0 = time.time()
    vol = tifffile.imread(tiff_path)
    print(f"TIFF loaded in {time.time()-t0:.2f}s. Shape: {vol.shape}, dtype: {vol.dtype}")
    D, H, W = vol.shape

    print("Computing XY Maximum Intensity Projection (MIP)...")
    # Subsample for percentile calculation
    p99 = np.percentile(vol[::2, ::4, ::4], 99.8)
    if p99 == 0:
        p99 = 1.0
    vol_capped = np.clip(vol, 0, p99).astype(np.float32) / float(p99)
    mip_xy = np.max(vol_capped, axis=0)

    print(f"Parsing SWC: {swc_path}")
    t_swc = time.time()
    nodes = {}
    edges = []
    trees = {} # nid -> tree_id
    
    with open(swc_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                nid = int(parts[0])
                ntype = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                r = float(parts[5])
                pid = int(parts[6])
                nodes[nid] = (x, y, z, ntype)
                if pid != -1:
                    edges.append((pid, nid))

    print(f"SWC parsed in {time.time()-t_swc:.2f}s. Loaded {len(nodes)} nodes, {len(edges)} edges.")

    # Physical scaling check
    rx, ry, rz = 0.1102, 0.1102, 0.5
    max_c = np.max([[n[0], n[1], n[2]] for n in nodes.values()], axis=0) if nodes else np.array([0,0,0])
    scale_factor = 1.0
    if max_c.max() < max(W, H) * rx * 1.5:
        scale_factor = 1.0 / rx

    print(f"Coordinate scale factor: {scale_factor:.4f}")

    # Build line segments
    lines = []
    line_types = []
    for pid, cid in edges:
        if pid in nodes and cid in nodes:
            px, py = nodes[pid][0] * scale_factor, nodes[pid][1] * scale_factor
            cx, cy = nodes[cid][0] * scale_factor, nodes[cid][1] * scale_factor
            lines.append([(px, py), (cx, cy)])
            line_types.append(nodes[cid][3])

    print(f"Generated {len(lines)} 2D line segments.")

    # Create multi-panel visualization: Full view + 2 zoom-ins
    fig = plt.figure(figsize=(20, 10), facecolor='#0d1117')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1], wspace=0.08, hspace=0.12)
    
    # Main Overview panel (Left, spanning both rows)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(mip_xy, cmap='magma', origin='upper', extent=[0, W, H, 0])
    
    # Overlay lines with downsampling if massive
    plot_lines = lines
    if len(plot_lines) > 250000:
        import random
        random.seed(42)
        idx_sample = random.sample(range(len(plot_lines)), 250000)
        plot_lines = [plot_lines[i] for i in idx_sample]
        print(f"Downsampled overlay to {len(plot_lines)} lines for rendering performance.")

    lc_main = LineCollection(plot_lines, colors='#00FFFF', linewidths=0.7, alpha=0.85)
    ax_main.add_collection(lc_main)
    ax_main.set_title(
        f"Full Volume XY Maximum Intensity Projection\nOverlay: Final Smoothed Neurons ({len(nodes):,} nodes, {len(edges):,} edges)",
        color='white', fontsize=14, pad=10, fontweight='bold'
    )
    ax_main.axis('off')

    # Calculate bounding box of neuron density to find interesting zoom crops
    if lines:
        all_pts = np.array([[l[0][0], l[0][1]] for l in lines])
        # Find two centroid clusters or dense regions
        center_x, center_y = np.median(all_pts[:, 0]), np.median(all_pts[:, 1])
        
        # Region 1: Center Region
        w_box, h_box = min(W // 3, 600), min(H // 3, 600)
        x1_min = max(0, int(center_x - w_box // 2))
        x1_max = min(W, x1_min + w_box)
        y1_min = max(0, int(center_y - h_box // 2))
        y1_max = min(H, y1_min + h_box)

        # Region 2: High intensity / high density region
        y_coords = (all_pts[:, 1]).astype(int)
        x_coords = (all_pts[:, 0]).astype(int)
        valid = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
        if np.any(valid):
            p_x = all_pts[valid, 0]
            p_y = all_pts[valid, 1]
            q75_x, q75_y = np.percentile(p_x, 75), np.percentile(p_y, 75)
            x2_min = max(0, int(q75_x - w_box // 2))
            x2_max = min(W, x2_min + w_box)
            y2_min = max(0, int(q75_y - h_box // 2))
            y2_max = min(H, y2_min + h_box)
        else:
            x2_min, x2_max, y2_min, y2_max = 0, w_box, 0, h_box

        # Zoom 1
        ax_z1 = fig.add_subplot(gs[0, 1])
        ax_z1.imshow(mip_xy[y1_min:y1_max, x1_min:x1_max], cmap='magma', origin='upper', extent=[x1_min, x1_max, y1_max, y1_min])
        lines_z1 = [l for l in lines if (x1_min <= l[0][0] <= x1_max or x1_min <= l[1][0] <= x1_max) and (y1_min <= l[0][1] <= y1_max or y1_min <= l[1][1] <= y1_max)]
        if lines_z1:
            lc_z1 = LineCollection(lines_z1, colors='#39FF14', linewidths=1.2, alpha=0.95)
            ax_z1.add_collection(lc_z1)
        ax_z1.set_title(f"Detailed Zoom Region A (X:[{x1_min}, {x1_max}], Y:[{y1_min}, {y1_max}])", color='white', fontsize=12, pad=6)
        ax_z1.axis('off')
        # Add box to main ax
        rect1 = plt.Rectangle((x1_min, y1_min), x1_max-x1_min, y1_max-y1_min, linewidth=1.5, edgecolor='#39FF14', facecolor='none', linestyle='--')
        ax_main.add_patch(rect1)

        # Zoom 2
        ax_z2 = fig.add_subplot(gs[1, 1])
        ax_z2.imshow(mip_xy[y2_min:y2_max, x2_min:x2_max], cmap='magma', origin='upper', extent=[x2_min, x2_max, y2_max, y2_min])
        lines_z2 = [l for l in lines if (x2_min <= l[0][0] <= x2_max or x2_min <= l[1][0] <= x2_max) and (y2_min <= l[0][1] <= y2_max or y2_min <= l[1][1] <= y2_max)]
        if lines_z2:
            lc_z2 = LineCollection(lines_z2, colors='#FFD700', linewidths=1.2, alpha=0.95)
            ax_z2.add_collection(lc_z2)
        ax_z2.set_title(f"Detailed Zoom Region B (X:[{x2_min}, {x2_max}], Y:[{y2_min}, {y2_max}])", color='white', fontsize=12, pad=6)
        ax_z2.axis('off')
        # Add box to main ax
        rect2 = plt.Rectangle((x2_min, y2_min), x2_max-x2_min, y2_max-y2_min, linewidth=1.5, edgecolor='#FFD700', facecolor='none', linestyle=':')
        ax_main.add_patch(rect2)

    plt.tight_layout()
    plt.savefig(output_png, facecolor='#0d1117', dpi=300)
    print(f"Visualization saved to {output_png}")
    
    if artifact_png:
        os.makedirs(os.path.dirname(artifact_png), exist_ok=True)
        import shutil
        shutil.copyfile(output_png, artifact_png)
        print(f"Copied artifact to {artifact_png}")

if __name__ == "__main__":
    render_visualization()
