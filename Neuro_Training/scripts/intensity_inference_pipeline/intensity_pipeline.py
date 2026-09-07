import os
import sys
import json
import torch
import numpy as np
import argparse
import tifffile
import time
from scipy.interpolate import splprep, splev
import concurrent.futures
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from Neuro_Training.scripts.inference import GapBridgingInferenceEngine

def setup_progress_logger(output_dir):
    progress_log_path = os.path.join(output_dir, 'pipeline_progress.log')
    plogger = logging.getLogger("ProgressTracker")
    plogger.setLevel(logging.INFO)
    plogger.propagate = False
    if plogger.hasHandlers(): plogger.handlers.clear()
    
    fh = logging.FileHandler(progress_log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    plogger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    plogger.addHandler(ch)
    return plogger, progress_log_path

class GPUVolumeProcessor:
    def __init__(self, device='cuda', plogger=None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.plogger = plogger

    def chunked_process(self, volume, func, chunk_size=(128, 512, 512), overlap=(0, 32, 32), desc="Processing"):
        D, H, W = volume.shape
        out_volume = np.zeros_like(volume, dtype=np.float32)
        
        cd, ch, cw = chunk_size
        od, oh, ow = overlap
        
        z_ranges = list(range(0, D, cd - od))
        y_ranges = list(range(0, H, ch - oh))
        x_ranges = list(range(0, W, cw - ow))
        
        total_chunks = len(z_ranges) * len(y_ranges) * len(x_ranges)
        chunk_idx = 0
        
        if self.plogger:
            self.plogger.info(f"--- Starting {desc} on {total_chunks} GPU Chunks ---")
            
        for z in z_ranges:
            for y in y_ranges:
                for x in x_ranges:
                    chunk_idx += 1
                    if self.plogger and (chunk_idx % 10 == 0 or chunk_idx == total_chunks or chunk_idx == 1):
                        self.plogger.info(f"[{desc}] Processing Chunk {chunk_idx}/{total_chunks} at Z:{z} Y:{y} X:{x}")
                    
                    z_end, y_end, x_end = min(D, z + cd), min(H, y + ch), min(W, x + cw)
                    chunk = volume[z:z_end, y:y_end, x:x_end]
                    t_chunk = torch.tensor(chunk.astype(np.float32), device=self.device).unsqueeze(0).unsqueeze(0)
                    
                    with torch.no_grad():
                        res_chunk = func(t_chunk)
                        
                    res_np = res_chunk.squeeze().cpu().numpy()
                    bz = od // 2 if z > 0 else 0
                    by = oh // 2 if y > 0 else 0
                    bx = ow // 2 if x > 0 else 0
                    ez = res_np.shape[0] - (od // 2 if z_end < D else 0)
                    ey = res_np.shape[1] - (oh // 2 if y_end < H else 0)
                    ex = res_np.shape[2] - (ow // 2 if x_end < W else 0)
                    
                    out_volume[z+bz:z_end-(res_np.shape[0]-ez), 
                               y+by:y_end-(res_np.shape[1]-ey), 
                               x+bx:x_end-(res_np.shape[2]-ex)] = np.maximum(
                                   out_volume[z+bz:z_end-(res_np.shape[0]-ez), 
                                              y+by:y_end-(res_np.shape[1]-ey), 
                                              x+bx:x_end-(res_np.shape[2]-ex)],
                                   res_np[bz:ez, by:ey, bx:ex]
                               )
        return out_volume

class MorphologicalSomaMasker:
    def __init__(self, resolution, erosion_radius_um=6.0, threshold=500.0, edge_threshold=400.0):
        self.res = np.array(resolution)
        self.erosion_radius_um = erosion_radius_um
        self.threshold = threshold
        self.edge_threshold = edge_threshold
        
    def get_masking_func(self):
        def gpu_morph_soma_mask(t_chunk):
            import torch.nn.functional as F
            
            # 1. Downsample (scale by 4 in XY, 2 in Z) for fast morphological operations
            scale_xy, scale_z = 4, 2
            kz = min(scale_z, t_chunk.shape[2])
            ky = min(scale_xy, t_chunk.shape[3])
            kx = min(scale_xy, t_chunk.shape[4])
            t_down = F.avg_pool3d(t_chunk, kernel_size=(kz, ky, kx), stride=(kz, ky, kx))
            
            # Calculate kernel size in downsampled space for Z, Y, X independently
            # Force Z kernel to 1 (no Z-erosion) to prevent erasing somas that are flattened in Z
            r_z = 0
            r_y = max(1, int(self.erosion_radius_um / (self.res[1] * scale_xy)))
            r_x = max(1, int(self.erosion_radius_um / (self.res[0] * scale_xy)))
            
            k_z = 1
            k_y = r_y * 2 + 1
            k_x = r_x * 2 + 1
            
            # 2. Morphological Erosion in XY (MinPool -> negative MaxPool on negative image)
            eroded = -F.max_pool3d(-t_down, kernel_size=(k_z, k_y, k_x), stride=1, padding=(r_z, r_y, r_x))
            
            # 3. Morphological Dilation in XY (MaxPool) on eroded image -> "Opening"
            opened = F.max_pool3d(eroded, kernel_size=(k_z, k_y, k_x), stride=1, padding=(r_z, r_y, r_x))
            
            # 4. Threshold to find thick soma cores (dendrites are destroyed by opening)
            soma_cores_down = (opened > self.threshold).float()
            
            # 5. Upsample soma cores back to raw chunk size
            soma_cores_up = F.interpolate(soma_cores_down, size=t_chunk.shape[2:], mode='nearest')
            
            # 6. Snap to edges: Dilate the cores slightly to create a search halo
            search_halo = F.max_pool3d(soma_cores_up, kernel_size=(3, 11, 11), stride=1, padding=(1, 5, 5))
            
            # 7. Final mask is intersection of search halo and raw intensity boundaries
            tight_mask = search_halo * (t_chunk > self.edge_threshold).float()
            
            # 8. Dilate the final tight mask slightly to ensure no jagged edges
            final_mask = F.max_pool3d(tight_mask, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
            
            return t_chunk * (1.0 - final_mask)
            
        return gpu_morph_soma_mask

class MultiScaleHessianRidgeFilter:
    def __init__(self, sigmas=[1.0, 3.0]):
        self.sigmas = sigmas
        
    def get_ridge_func(self):
        def gpu_hessian_ridge(t_chunk):
            import torch.nn.functional as F
            device = t_chunk.device
            max_ridge = torch.zeros_like(t_chunk)
            
            for sigma in self.sigmas:
                k = max(3, int(sigma * 3) * 2 + 1)
                pad = k // 2
                x = torch.arange(-pad, pad+1, dtype=torch.float32, device=device)
                g1 = torch.exp(-(x**2)/(2*sigma**2))
                g1 = g1 / g1.sum()
                g2 = torch.exp(-(x**2)/(2*(sigma*1.5)**2))
                g2 = g2 / g2.sum()
                
                w1_x, w1_y, w1_z = g1.view(1, 1, 1, 1, -1), g1.view(1, 1, 1, -1, 1), g1.view(1, 1, -1, 1, 1)
                c = F.conv3d(t_chunk, w1_x, padding=(0, 0, pad))
                c = F.conv3d(c, w1_y, padding=(0, pad, 0))
                smooth1 = F.conv3d(c, w1_z, padding=(pad, 0, 0))
                
                w2_x, w2_y, w2_z = g2.view(1, 1, 1, 1, -1), g2.view(1, 1, 1, -1, 1), g2.view(1, 1, -1, 1, 1)
                c2 = F.conv3d(t_chunk, w2_x, padding=(0, 0, pad))
                c2 = F.conv3d(c2, w2_y, padding=(0, pad, 0))
                smooth2 = F.conv3d(c2, w2_z, padding=(pad, 0, 0))
                
                dog = torch.relu(smooth1 - smooth2)
                max_ridge = torch.max(max_ridge, dog)
                
            return max_ridge
        return gpu_hessian_ridge

class RidgeBasedSeeder:
    def __init__(self, plogger=None, engine=None, resolution=(0.1102, 0.1102, 0.5)):
        self.plogger = plogger
        self.engine = engine
        self.res = resolution
        
    def extract_fragments(self, ridge_volume, output_path_pre, output_path_post, raw_tiff_path, ridge_threshold=50.0, min_seed_length_um=20.0):
        import collections
        from joblib import Parallel, delayed
        import multiprocessing as mp
        from skimage.morphology import skeletonize
        
        self.plogger.info(f"Extracting ridge fragments. Ridge map threshold: {ridge_threshold}")
        t0 = time.time()
        binary_vol = (ridge_volume > ridge_threshold).astype(np.uint8)
        
        self.plogger.info("Running chunked 3D Skeletonization on Ridge Map...")
        D, H, W = binary_vol.shape
        cd, ch, cw = 271, 512, 512
        od, oh, ow = 0, 32, 32
        
        z_ranges = list(range(0, D, cd - od)) if cd - od > 0 else [0]
        y_ranges = list(range(0, H, ch - oh)) if ch - oh > 0 else [0]
        x_ranges = list(range(0, W, cw - ow)) if cw - ow > 0 else [0]
        
        tasks = []
        for z in z_ranges:
            for y in y_ranges:
                for x in x_ranges:
                    tasks.append((z, min(D, z + cd), y, min(H, y + ch), x, min(W, x + cw)))
                    
        def process_chunk(coords):
            z, z_end, y, y_end, x, x_end = coords
            chunk = binary_vol[z:z_end, y:y_end, x:x_end]
            return coords, skeletonize(chunk) if chunk.max() > 0 else np.zeros_like(chunk)
                
        results = Parallel(n_jobs=mp.cpu_count(), backend='loky')(delayed(process_chunk)(coords) for coords in tasks)
        
        skeleton = np.zeros_like(binary_vol)
        for coords, skel_chunk in results:
            z, z_end, y, y_end, x, x_end = coords
            bz = od // 2 if z > 0 else 0
            by = oh // 2 if y > 0 else 0
            bx = ow // 2 if x > 0 else 0
            ez = skel_chunk.shape[0] - (od // 2 if z_end < D else 0)
            ey = skel_chunk.shape[1] - (oh // 2 if y_end < H else 0)
            ex = skel_chunk.shape[2] - (ow // 2 if x_end < W else 0)
            skeleton[z+bz:z_end-(skel_chunk.shape[0]-ez), y+by:y_end-(skel_chunk.shape[1]-ey), x+bx:x_end-(skel_chunk.shape[2]-ex)] |= skel_chunk[bz:ez, by:ey, bx:ex]
                     
        self.plogger.info("Extracting graph & applying Biological Plausibility Filter...")
        pts = np.argwhere(skeleton > 0)
        node_ids = np.zeros_like(skeleton, dtype=np.int32)
        for i, pt in enumerate(pts): node_ids[pt[0], pt[1], pt[2]] = i + 1
            
        valid_pre, valid_post, bio_dropped, size_dropped = 0, 0, 0, 0
        
        with open(output_path_pre, 'w') as f_pre, open(output_path_post, 'w') as f_post:
            f_pre.write("# Unsupervised Ridge Seeds (Pre-Bio Geometric)\n")
            f_post.write("# Unsupervised Ridge Seeds (Biologically Validated)\n")
            visited = set()
            D_s, H_s, W_s = skeleton.shape
            
            pts_processed = 0
            total_pts = len(pts)
            
            for pt in pts:
                pts_processed += 1
                if pts_processed % 10000 == 0:
                    self.plogger.info(f"Graph Extraction Progress: {pts_processed}/{total_pts} total skeleton points scanned...")
                    
                pt_tuple = tuple(pt)
                if pt_tuple in visited: continue
                
                visited.add(pt_tuple)
                queue = collections.deque([(pt, -1)])
                component_nodes = []
                inner_pts_processed = 0
                
                while queue:
                    curr_pt, parent_nid = queue.popleft()
                    inner_pts_processed += 1
                    
                    if inner_pts_processed % 500000 == 0:
                        self.plogger.info(f"    -> Still traversing massive component... extracted {inner_pts_processed} connected nodes so far.")
                    
                    curr_nid = node_ids[curr_pt[0], curr_pt[1], curr_pt[2]]
                    component_nodes.append((curr_nid, curr_pt, parent_nid))
                    
                    z_c, y_c, x_c = curr_pt
                    for dz in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dz == 0 and dy == 0 and dx == 0: continue
                                nz, ny, nx = z_c + dz, y_c + dy, x_c + dx
                                if 0 <= nz < D_s and 0 <= ny < H_s and 0 <= nx < W_s:
                                    if skeleton[nz, ny, nx] > 0 and (nz, ny, nx) not in visited:
                                        visited.add((nz, ny, nx))
                                        queue.append(((nz, ny, nx), curr_nid))
                
                # 1. Filter by physical size
                approx_length = len(component_nodes) * np.mean(self.res)
                if approx_length < min_seed_length_um:
                    size_dropped += 1
                    continue
                    
                # Write to Pre-Bio file (Geometric only)
                valid_pre += 1
                if valid_pre % 50 == 0:
                    self.plogger.info(f"Geometrically extracted fragment {valid_pre}...")
                    
                for nid, c_pt, pid in component_nodes:
                    f_pre.write(f"{nid} 3 {c_pt[2]:.2f} {c_pt[1]:.2f} {c_pt[0]:.2f} 1.0 {pid}\n")
        
        self.plogger.info(f"Graph extraction complete. Geometrically valid: {valid_pre}.")
        self.plogger.info(f"Dropped {size_dropped} small specks.")
        
        # 3. Generative Bridging & Validation Phase (LLM)
        if self.engine is not None:
            self.plogger.info("Invoking Bio-Tower Generative Engine to merge fragments and extend open paths...")
            self.engine.bridge_and_connect_swc(
                input_swc_path=output_path_pre,
                output_swc_path=output_path_post,
                tiff_volume_path=raw_tiff_path,
                physical_resolution=self.res
            )
            return output_path_post
            
        return output_path_pre

class SplineSmoother:
    def smooth_swc(self, input_swc, output_swc, density_voxels=10.0):
        # Implementation remains the same as before
        nodes = {}
        with open(input_swc, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    p = line.strip().split()
                    if len(p) >= 7:
                        nodes[int(p[0])] = {
                            'id': int(p[0]), 'type': int(p[1]),
                            'x': float(p[2]), 'y': float(p[3]), 'z': float(p[4]),
                            'r': float(p[5]), 'pid': int(p[6])
                        }
        
        children = {}
        for nid, n in nodes.items():
            children.setdefault(n['pid'], []).append(nid)
            
        critical_nodes = set([-1])
        for nid, n in nodes.items():
            if n['pid'] == -1 or len(children.get(nid, [])) != 1:
                critical_nodes.add(nid)
                
        segments = []
        visited = set()
        for nid in nodes:
            if nid in critical_nodes and nid not in visited:
                for child_id in children.get(nid, []):
                    seg = [nid]
                    curr = child_id
                    while curr not in critical_nodes:
                        seg.append(curr)
                        visited.add(curr)
                        if not children.get(curr): break
                        curr = children[curr][0]
                    seg.append(curr)
                    segments.append(seg)
                    
        smoothed_nodes = []
        new_id = 1
        id_map = {}
        
        for seg in segments:
            pts = np.array([[nodes[n]['x'], nodes[n]['y'], nodes[n]['z']] for n in seg])
            if len(pts) >= 4:
                try:
                    tck, u = splprep([pts[:,0], pts[:,1], pts[:,2]], s=3.0)
                    arc_len = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                    num_pts = max(3, int(arc_len / density_voxels))
                    u_new = np.linspace(0, 1, num_pts)
                    x_new, y_new, z_new = splev(u_new, tck)
                    
                    prev_id = id_map.get(seg[0], -1) if id_map.get(seg[0], -1) != -1 else nodes[seg[0]]['pid']
                    for i in range(len(x_new)):
                        smoothed_nodes.append({
                            'id': new_id, 'type': nodes[seg[0]]['type'],
                            'x': x_new[i], 'y': y_new[i], 'z': z_new[i],
                            'r': nodes[seg[0]]['r'], 'pid': prev_id
                        })
                        if i == 0: id_map[seg[0]] = new_id
                        if i == len(x_new) - 1: id_map[seg[-1]] = new_id
                        prev_id = new_id
                        new_id += 1
                except:
                    prev_id = id_map.get(seg[0], -1) if id_map.get(seg[0], -1) != -1 else nodes[seg[0]]['pid']
                    for nid in seg:
                        n = nodes[nid]
                        smoothed_nodes.append({
                            'id': new_id, 'type': n['type'],
                            'x': n['x'], 'y': n['y'], 'z': n['z'],
                            'r': n['r'], 'pid': prev_id
                        })
                        id_map[nid] = new_id
                        prev_id = new_id
                        new_id += 1
            else:
                prev_id = id_map.get(seg[0], -1) if id_map.get(seg[0], -1) != -1 else nodes[seg[0]]['pid']
                for nid in seg:
                    n = nodes[nid]
                    smoothed_nodes.append({
                        'id': new_id, 'type': n['type'],
                        'x': n['x'], 'y': n['y'], 'z': n['z'],
                        'r': n['r'], 'pid': prev_id
                    })
                    id_map[nid] = new_id
                    prev_id = new_id
                    new_id += 1
                    
        with open(output_swc, 'w') as f:
            for n in smoothed_nodes:
                f.write(f"{n['id']} {n['type']} {n['x']:.2f} {n['y']:.2f} {n['z']:.2f} {n['r']:.2f} {n['pid']}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff_volume', required=True)
    parser.add_argument('--output_dir', default='intensity_pipeline_outputs')
    parser.add_argument('--resolution', type=float, nargs=3, default=[0.1102, 0.1102, 0.5])
    parser.add_argument('--ridge_threshold', type=float, default=50.0, help="Absolute response cutoff for DoG ridge detection")
    parser.add_argument('--min_seed_length_um', type=float, default=20.0)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    plogger, log_path = setup_progress_logger(args.output_dir)
    
    plogger.info("===== Starting Intensity Inference Pipeline Overhaul =====")
    
    t_start = time.time()
    vol = tifffile.imread(args.tiff_volume)
    plogger.info(f"1. Volume loaded. Shape: {vol.shape}, Dtype: {vol.dtype}")
    
    masked_tiff_path = os.path.join(args.output_dir, "masked_volume.tif")
    if os.path.exists(masked_tiff_path):
        plogger.info("2. Found cached masked volume, loading directly...")
        masked_vol = tifffile.imread(masked_tiff_path)
    else:
        plogger.info("2. Morphological Soma Masking (2D XY Opening & Edge Snapping)...")
        gpu_proc = GPUVolumeProcessor(plogger=plogger)
        masker = MorphologicalSomaMasker(args.resolution, erosion_radius_um=3.5, threshold=150.0, edge_threshold=100.0)
        masked_vol = gpu_proc.chunked_process(vol, masker.get_masking_func(), desc="Soma Masking")
        tifffile.imwrite(masked_tiff_path, masked_vol.astype(vol.dtype))
        
    ridge_map_path = os.path.join(args.output_dir, "ridge_map_volume.tif")
    if os.path.exists(ridge_map_path):
        plogger.info("3. Found cached Ridge Map, loading directly...")
        ridge_vol = tifffile.imread(ridge_map_path)
    else:
        plogger.info("3. Multi-Scale Hessian Ridge Detection (DoG bandpass)...")
        gpu_proc = GPUVolumeProcessor(plogger=plogger)
        ridge_filter = MultiScaleHessianRidgeFilter(sigmas=[1.0, 3.0])
        ridge_vol = gpu_proc.chunked_process(masked_vol, ridge_filter.get_ridge_func(), chunk_size=(128, 512, 512), desc="Ridge Extraction")
        tifffile.imwrite(ridge_map_path, ridge_vol.astype(vol.dtype))
        
    plogger.info("4. Initializing Bio-Tower Generative Engine...")
    config_path = os.path.join(os.path.dirname(__file__), '../../../Neuro_Tokenization/config.json')
    model_config_path = os.path.join(os.path.dirname(__file__), '../../../Neuro_Model/config.json')
    with open(config_path, 'r') as f: train_config = json.load(f)
    with open(model_config_path, 'r') as f: model_config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    default_ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../checkpoints/checkpoint_epoch_29.pt'))
    engine = GapBridgingInferenceEngine(model_config, train_config.get('inference_parameters', {}), device, checkpoint_path=default_ckpt)
        
    plogger.info("5. Unsupervised Ridge Skeleton Seeding & Bio-Plausibility Filtering...")
    seeder = RidgeBasedSeeder(plogger=plogger, engine=engine, resolution=tuple(args.resolution))
    seed_swc_pre = os.path.join(args.output_dir, "unsupervised_seeds_pre_bio.swc")
    seed_swc_post = os.path.join(args.output_dir, "unsupervised_seeds_post_bio.swc")
    # For testing, we overwrite previous seeds
    if os.path.exists(seed_swc_pre): os.remove(seed_swc_pre)
    if os.path.exists(seed_swc_post): os.remove(seed_swc_post)
    seeder.extract_fragments(ridge_vol, seed_swc_pre, seed_swc_post, raw_tiff_path=args.tiff_volume, ridge_threshold=args.ridge_threshold, min_seed_length_um=args.min_seed_length_um)
    
    plogger.info("6. Ridge-Weighted Gap Bridging & Biological Extension...")
    engine.preprocessor._cached_volume = ridge_vol
    engine.preprocessor._cached_tiff_path = args.tiff_volume
    raw_output_swc = os.path.join(args.output_dir, "raw_generative_inference.swc")
    plogger.info("   -> Endpoints will naturally trace fluctuating ridges in the Bio-Tower.")
    
    engine.bridge_and_connect_swc(
        seed_swc_post, raw_output_swc, args.tiff_volume, 
        physical_resolution=tuple(args.resolution),
        max_gap_distance_um=40.0, bio_bridge_threshold=0.15
    )
    
    plogger.info("7. Scaling to Physical Coordinates (Micrometers)...")
    res_x, res_y, res_z = args.resolution
    
    def scale_swc(in_swc, out_swc):
        import pandas as pd
        if os.path.exists(in_swc):
            df = pd.read_csv(in_swc, sep=r'\s+', comment='#', header=None, names=['id', 'type', 'x', 'y', 'z', 'r', 'pid'])
            df['x'] = df['x'] * res_x
            df['y'] = df['y'] * res_y
            df['z'] = df['z'] * res_z
            df.to_csv(out_swc, sep=' ', index=False, header=False, float_format='%.3f')
            
    final_swc = os.path.join(args.output_dir, "final.swc")
    seed_swc_post_scaled = os.path.join(args.output_dir, "unsupervised_seeds_post_bio_scaled.swc")
    
    scale_swc(raw_output_swc, final_swc)
    scale_swc(seed_swc_post, seed_swc_post_scaled)
    plogger.info(f"   -> Final scaled SWC saved to {final_swc}")
    
    plogger.info(f"===== Pipeline Overhaul Complete in {time.time()-t_start:.2f}s =====")
    plogger.info(f"Detailed log saved to {log_path}")

if __name__ == "__main__":
    main()
