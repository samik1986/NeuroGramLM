"""
Visualizes SWC skeletons overlaid on 3D TIFF intensity volumes.
Generates:
1. Multi-view Maximum Intensity Projections (MIP) with SWC traces in XY, XZ, YZ planes.
2. 3D Volume skeleton trajectory plot with intensity context.
3. High-resolution local zoom-in MIP showing dense neural arborization and ridge alignment.
"""

import os
import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_swc(swc_path):
    nodes = {}
    edges = []
    with open(swc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                nid = int(parts[0])
                ntype = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                r = float(parts[5])
                pid = int(parts[6])
                nodes[nid] = (x, y, z, r, ntype, pid)
                if pid != -1:
                    edges.append((pid, nid))
    return nodes, edges

def visualize_swc_on_volume(
    tiff_path, 
    swc_path, 
    output_png="swc_volume_visualization.png",
    resolution=(0.112, 0.1102, 0.5) # rx, ry, rz in microns/px
):
    print(f"Loading TIFF volume: {tiff_path}")
    vol = tifffile.imread(tiff_path)
    D, H, W = vol.shape
    print(f"TIFF Shape: {vol.shape}, Dtype: {vol.dtype}")
    
    rx, ry, rz = resolution
    nodes, edges = load_swc(swc_path)
    print(f"Loaded {len(nodes)} nodes, {len(edges)} connections from {swc_path}")
    
    # Convert SWC coordinates (microns) to Voxel indices (X_px, Y_px, Z_px)
    voxel_coords = {}
    for nid, (x_um, y_um, z_um, r, ntype, pid) in nodes.items():
        x_px = x_um / rx
        y_px = y_um / ry
        z_px = z_um / rz
        voxel_coords[nid] = (x_px, y_px, z_px)
        
    all_px = np.array(list(voxel_coords.values()))
    min_x, max_x = np.clip(np.min(all_px[:, 0]), 0, W-1), np.clip(np.max(all_px[:, 0]), 0, W-1)
    min_y, max_y = np.clip(np.min(all_px[:, 1]), 0, H-1), np.clip(np.max(all_px[:, 1]), 0, H-1)
    min_z, max_z = np.clip(np.min(all_px[:, 2]), 0, D-1), np.clip(np.max(all_px[:, 2]), 0, D-1)
    
    print(f"Skeleton Voxel Bounding Box: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}], Z=[{min_z:.1f}, {max_z:.1f}]")
    
    # Crop subvolume around the skeleton region with padding
    pad_xy, pad_z = 50, 10
    crop_x1, crop_x2 = int(max(0, min_x - pad_xy)), int(min(W, max_x + pad_xy))
    crop_y1, crop_y2 = int(max(0, min_y - pad_xy)), int(min(H, max_y + pad_xy))
    crop_z1, crop_z2 = int(max(0, min_z - pad_z)), int(min(D, max_z + pad_z))
    
    subvol = vol[crop_z1:crop_z2, crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
    # Background subtraction / percentile contrast stretch
    p1, p99 = np.percentile(subvol, 5), np.percentile(subvol, 99.8)
    subvol = np.clip((subvol - p1) / (p99 - p1 + 1e-6), 0, 1)
    
    # Compute Maximum Intensity Projections (MIP)
    mip_xy = np.max(subvol, axis=0) # (H_sub, W_sub)
    mip_xz = np.max(subvol, axis=1) # (D_sub, W_sub)
    
    # Setup multi-panel plot
    fig = plt.figure(figsize=(18, 12), dpi=150, facecolor='#0d1117')
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    
    # 1. 3D Skeleton + Volume Cloud
    ax3d = fig.add_subplot(2, 2, 1, projection='3d', facecolor='#0d1117')
    ax3d.set_title("3D Neural Skeleton in Physical Space (Microns)", color='#58a6ff', fontsize=12, fontweight='bold', pad=12)
    
    # Downsample edges for clean 3D rendering
    sample_stride = max(1, len(edges) // 4000)
    for p_id, c_id in edges[::sample_stride]:
        if p_id in nodes and c_id in nodes:
            x_pair = [nodes[p_id][0], nodes[c_id][0]]
            y_pair = [nodes[p_id][1], nodes[c_id][1]]
            z_pair = [nodes[p_id][2], nodes[c_id][2]]
            ax3d.plot(x_pair, y_pair, z_pair, color='#38ef7d', alpha=0.5, linewidth=0.7)
            
    # Highlight endpoints / roots
    root_nodes = [nid for nid, val in nodes.items() if val[5] == -1]
    rx_pts = [nodes[nid][0] for nid in root_nodes]
    ry_pts = [nodes[nid][1] for nid in root_nodes]
    rz_pts = [nodes[nid][2] for nid in root_nodes]
    ax3d.scatter(rx_pts, ry_pts, rz_pts, color='#ff0844', s=15, label='Fragment Roots / Joint Anchors', alpha=0.9)
    
    ax3d.set_xlabel('X (μm)', color='white', fontsize=9)
    ax3d.set_ylabel('Y (μm)', color='white', fontsize=9)
    ax3d.set_zlabel('Z (μm)', color='white', fontsize=9)
    ax3d.tick_params(colors='white', labelsize=8)
    ax3d.grid(True, linestyle=':', alpha=0.2)
    ax3d.view_init(elev=28, azim=-55)
    ax3d.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=8)
    
    # 2. XY Plane Maximum Intensity Projection (MIP)
    ax_xy = fig.add_subplot(2, 2, 2, facecolor='#0d1117')
    ax_xy.set_title("XY Plane (MIP) Intensity & SWC Overlay", color='#58a6ff', fontsize=12, fontweight='bold')
    ax_xy.imshow(mip_xy, cmap='magma', extent=[crop_x1, crop_x2, crop_y2, crop_y1], origin='upper', alpha=0.9)
    
    for p_id, c_id in edges[::sample_stride]:
        if p_id in voxel_coords and c_id in voxel_coords:
            xp = [voxel_coords[p_id][0], voxel_coords[c_id][0]]
            yp = [voxel_coords[p_id][1], voxel_coords[c_id][1]]
            ax_xy.plot(xp, yp, color='#00ffcc', alpha=0.7, linewidth=0.9)
    ax_xy.set_xlabel('X (voxels)', color='white', fontsize=9)
    ax_xy.set_ylabel('Y (voxels)', color='white', fontsize=9)
    ax_xy.tick_params(colors='white', labelsize=8)
    ax_xy.set_xlim(crop_x1, crop_x2)
    ax_xy.set_ylim(crop_y2, crop_y1)
    
    # 3. XZ Plane Maximum Intensity Projection (MIP)
    ax_xz = fig.add_subplot(2, 2, 3, facecolor='#0d1117')
    ax_xz.set_title("XZ Plane (MIP) Depth Profile & SWC Overlay", color='#58a6ff', fontsize=12, fontweight='bold')
    ax_xz.imshow(mip_xz, cmap='magma', extent=[crop_x1, crop_x2, crop_z2, crop_z1], origin='upper', aspect=3.0, alpha=0.9)
    
    for p_id, c_id in edges[::sample_stride]:
        if p_id in voxel_coords and c_id in voxel_coords:
            xp = [voxel_coords[p_id][0], voxel_coords[c_id][0]]
            zp = [voxel_coords[p_id][2], voxel_coords[c_id][2]]
            ax_xz.plot(xp, zp, color='#00ffcc', alpha=0.7, linewidth=0.9)
    ax_xz.set_xlabel('X (voxels)', color='white', fontsize=9)
    ax_xz.set_ylabel('Z (optical slices)', color='white', fontsize=9)
    ax_xz.tick_params(colors='white', labelsize=8)
    ax_xz.set_xlim(crop_x1, crop_x2)
    ax_xz.set_ylim(crop_z2, crop_z1)
    
    # 4. Dense High-Resolution Region of Interest (ROI)
    ax_roi = fig.add_subplot(2, 2, 4, facecolor='#0d1117')
    ax_roi.set_title("Local High-Res Zoom: Filament & Predicted Gap Bridge", color='#58a6ff', fontsize=12, fontweight='bold')
    
    # Focus on center of skeleton mass
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    roi_r = 250
    rx1, rx2 = max(0, center_x - roi_r), min(W, center_x + roi_r)
    ry1, ry2 = max(0, center_y - roi_r), min(H, center_y + roi_r)
    
    # Crop ROI MIP
    roi_sub = vol[:, int(ry1):int(ry2), int(rx1):int(rx2)].astype(np.float32)
    p1_r, p99_r = np.percentile(roi_sub, 5), np.percentile(roi_sub, 99.8)
    roi_sub = np.clip((roi_sub - p1_r) / (p99_r - p1_r + 1e-6), 0, 1)
    roi_mip = np.max(roi_sub, axis=0)
    
    ax_roi.imshow(roi_mip, cmap='inferno', extent=[rx1, rx2, ry2, ry1], origin='upper')
    
    for p_id, c_id in edges:
        if p_id in voxel_coords and c_id in voxel_coords:
            p_coord = voxel_coords[p_id]
            c_coord = voxel_coords[c_id]
            if (rx1 <= p_coord[0] <= rx2 and ry1 <= p_coord[1] <= ry2) or \
               (rx1 <= c_coord[0] <= rx2 and ry1 <= c_coord[1] <= ry2):
                ax_roi.plot([p_coord[0], c_coord[0]], [p_coord[1], c_coord[1]], color='#39ff14', linewidth=1.3, alpha=0.9)
                
    ax_roi.set_xlim(rx1, rx2)
    ax_roi.set_ylim(ry2, ry1)
    ax_roi.set_xlabel('X (voxels)', color='white', fontsize=9)
    ax_roi.set_ylabel('Y (voxels)', color='white', fontsize=9)
    ax_roi.tick_params(colors='white', labelsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    plt.close()
    print(f"Visualization figure saved successfully to: {output_png}")

if __name__ == "__main__":
    visualize_swc_on_volume(
        tiff_path="F0046_multichannel_cmle_ch03.tif",
        swc_path="predicted_joined_skeleton.swc",
        output_png="swc_volume_visualization.png",
        resolution=(0.112, 0.1102, 0.5)
    )
