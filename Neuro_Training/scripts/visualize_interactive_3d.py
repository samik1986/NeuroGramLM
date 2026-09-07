import os
import time
import numpy as np
import tifffile
import plotly.graph_objects as go

def generate_isosurface_3d(
    raw_path="F0046_multichannel_cmle_ch03.tif",
    masked_path="intensity_pipeline_outputs/masked_volume.tif",
    ridge_path="intensity_pipeline_outputs/ridge_map_volume.tif",
    output_html="proper_3d_overlay.html",
    subsample_xy=32,
    subsample_z=4
):
    print("Loading TIFFs for 3D Isosurface Overlay...")
    if not all(os.path.exists(p) for p in [raw_path, masked_path, ridge_path]):
        print("Error: Missing TIFF volumes.")
        return
        
    raw_vol = tifffile.imread(raw_path)
    masked_vol = tifffile.imread(masked_path)
    ridge_vol = tifffile.imread(ridge_path)
    
    # Downsample heavily
    raw_sub = raw_vol[::subsample_z, ::subsample_xy, ::subsample_xy]
    masked_sub = masked_vol[::subsample_z, ::subsample_xy, ::subsample_xy]
    ridge_sub = ridge_vol[::subsample_z, ::subsample_xy, ::subsample_xy]
    
    D, H, W = raw_sub.shape
    
    # Create 3D grid coordinates
    Z, Y, X = np.mgrid[0:D, 0:H, 0:W]
    
    # Physical scaling for the coordinates
    res_z = 0.5 * subsample_z
    res_y = 0.1102 * subsample_xy
    res_x = 0.1102 * subsample_xy
    
    X_flat = (X.flatten() * res_x).astype(np.float32)
    Y_flat = (Y.flatten() * res_y).astype(np.float32)
    Z_flat = (Z.flatten() * res_z).astype(np.float32)
    
    traces = []
    
    # 1. Background as faint scatter to keep performance high
    # (Isosurfacing the entire background is very noisy and blocks the view)
    p95 = np.percentile(raw_sub, 95)
    bg_mask = (raw_sub > p95)
    bg_z, bg_y, bg_x = np.where(bg_mask)
    traces.append(go.Scatter3d(
        x=bg_x * res_x, y=bg_y * res_y, z=bg_z * res_z,
        mode='markers',
        marker=dict(size=1.5, color='lightgray', opacity=0.15),
        name='Raw Background'
    ))
    
    # 2. Soma Mask as an Isosurface
    # We define soma intensity where the raw is bright but masked is zero
    soma_intensity = np.where((masked_sub == 0), raw_sub, 0)
    soma_flat = soma_intensity.flatten()
    
    # Only render somas above a threshold to avoid meshing empty space
    soma_thresh = p95
    #if np.max(soma_flat) > soma_thresh:
    #    traces.append(go.Isosurface(
    #        x=X_flat, y=Y_flat, z=Z_flat,
    #        value=soma_flat,
    #        isomin=soma_thresh,
    #        isomax=np.max(soma_flat),
    #        surface_fill=1.0,
    #        opacity=0.6,
    #        colorscale=[[0, 'red'], [1, 'darkred']],
    #        showscale=False,
    #        name='Soma Isosurface'
    #    ))

    # 3. Ridge Map as an Isosurface (Solid Tubes)
    ridge_flat = ridge_sub.flatten()
    ridge_thresh = np.percentile(ridge_flat, 99.5)
    
    #if np.max(ridge_flat) > ridge_thresh:
    #    traces.append(go.Isosurface(
    #        x=X_flat, y=Y_flat, z=Z_flat,
    #        value=ridge_flat,
    #        isomin=ridge_thresh,
    #        isomax=np.max(ridge_flat),
    #        surface_fill=1.0,
    #        opacity=0.8,
    #        colorscale=[[0, 'cyan'], [1, 'blue']],
    #        showscale=False,
    #        name='Ridge Map Isosurface'
    #    ))

    # 4. Extracted Skeletons (Pre-LLM)
    base_swc = "intensity_pipeline_outputs/unsupervised_seeds_pre_bio.swc"
    base_edges = set()
    if os.path.exists(base_swc):
        import pandas as pd
        b_df = pd.read_csv(base_swc, sep=r'\s+', comment='#', header=None,
                             names=['id', 'type', 'x', 'y', 'z', 'r', 'pid'])
        b_dict = {row['id']: row for _, row in b_df.iterrows()}
        bx, by, bz = [], [], []
        for _, row in b_df.iterrows():
            pid = row['pid']
            if pid != -1:
                base_edges.add((int(row['id']), int(pid)))
            if pid in b_dict:
                parent = b_dict[pid]
                bx.extend([row['x']*0.1102, parent['x']*0.1102, None])
                by.extend([row['y']*0.1102, parent['y']*0.1102, None])
                bz.extend([row['z']*0.5, parent['z']*0.5, None])
        traces.append(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode='lines',
            line=dict(color='lime', width=4),
            name='Neurons from Detection Phase'
        ))

    # 5. Final Generative Bridged Skeletons (Lines)
    swc_path = "intensity_pipeline_outputs/raw_generative_inference.swc"
    if os.path.exists(swc_path):
        print(f"Overlaying generated skeletons from: {swc_path}")
        import pandas as pd
        # Read SWC
        swc_df = pd.read_csv(swc_path, sep=r'\s+', comment='#', header=None,
                             names=['id', 'type', 'x', 'y', 'z', 'r', 'pid'])
        
        # Build segments using pid mapping to draw discrete lines
        node_dict = {row['id']: row for _, row in swc_df.iterrows()}
        x_lines, y_lines, z_lines = [], [], []
        
        for _, row in swc_df.iterrows():
            pid = row['pid']
            child_id = int(row['id'])
            # ONLY draw the edge in yellow if this edge did not exist in the base graph!
            if pid != -1 and (child_id, int(pid)) not in base_edges:
                if pid in node_dict:
                    parent = node_dict[pid]
                    x_lines.extend([row['x']*0.1102, parent['x']*0.1102, None])
                    y_lines.extend([row['y']*0.1102, parent['y']*0.1102, None])
                    z_lines.extend([row['z']*0.5, parent['z']*0.5, None])
                
        traces.append(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color='yellow', width=5),
            name='LLM Generated Neurons'
        ))

    # Keep original aspect ratio
    max_dim = max(W*res_x, H*res_y, D*res_z)
    aspect_ratio = dict(x=(W*res_x)/max_dim, y=(H*res_y)/max_dim, z=(D*res_z)/max_dim)

    layout = go.Layout(
        title='Interactive 3D Isosurface Overlay (Smooth Volumes)',
        scene=dict(
            xaxis=dict(title='X (um)', showbackground=False),
            yaxis=dict(title='Y (um)', showbackground=False),
            zaxis=dict(title='Z (um)', showbackground=False),
            aspectmode='manual',
            aspectratio=aspect_ratio,
            bgcolor='black'
        ),
        paper_bgcolor='black',
        font=dict(color='white')
    )
    
    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(output_html)
    print(f"Successfully generated Isosurface HTML at: {output_html}")
    try:
        png_path = output_html.replace(".html", ".png")
        fig.write_image(png_path, width=1920, height=1080)
        print(f"Successfully generated Isosurface PNG at: {png_path}")
    except Exception as e:
        print(f"Failed to generate PNG: {e}")

if __name__ == "__main__":
    generate_isosurface_3d()
