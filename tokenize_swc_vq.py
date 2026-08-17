"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python tokenize_swc_vq.py
Input: A directory of .swc files specified by SWC_DIR in config.json
Output: A jsonl file containing tokenized sequences (default: tokenized_dataset.jsonl), and trained K-Means vocabulary models.
"""

import os
import sys
import numpy as np
from sklearn.cluster import KMeans
import json
import warnings
import random
import joblib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

with open('config.json', 'r') as f:
    config = json.load(f)

SWC_DIR = config.get("SWC_DIR", "SWCs/")
NUM_CLUSTERS = config.get("NUM_CLUSTERS", 512)
JUMP_BIN_SIZE = config.get("JUMP_BIN_SIZE", 10.0)
GEO_VOCAB_MODEL_PATH = config.get("GEO_VOCAB_MODEL_PATH", "kmeans_geo_vocab.pkl")
INV_VOCAB_MODEL_PATH = config.get("INV_VOCAB_MODEL_PATH", "kmeans_inv_vocab.pkl")
VOCAB_TRAIN_FILES = config.get("VOCAB_TRAIN_FILES", 1000)
MAX_SAMPLES = config.get("MAX_SAMPLES_FOR_KMEANS", 100000)
MAX_WORKERS = config.get("MAX_WORKERS", 16)
OUTPUT_FILE = config.get("OUTPUT_FILE", "tokenized_dataset.jsonl")

# Global atlas for workers
annotation_volume = None
resolution = 25

def init_worker():
    global annotation_volume, resolution
    try:
        import nrrd
        # The NRRD file downloaded by download_ccf.py
        data, header = nrrd.read('annotation_25.nrrd')
        annotation_volume = data
    except Exception as e:
        print(f"Worker atlas init failed: {e}")
        annotation_volume = None

def parse_swc(filepath):
    nodes = {}
    children = defaultdict(list)
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 7:
                node_id = int(parts[0])
                n_type = int(parts[1])
                x, y, z, r = map(float, parts[2:6])
                parent_id = int(parts[6])
                
                nodes[node_id] = (x, y, z, r, parent_id)
                if parent_id != -1 and parent_id in nodes:
                    children[parent_id].append(node_id)
                    
    all_children = set(child for child_list in children.values() for child in child_list)
    roots = [n for n in nodes.keys() if n not in all_children]
    return nodes, dict(children), roots

def get_inv_features(u, v, nodes):
    dx = nodes[v][0] - nodes[u][0]
    dy = nodes[v][1] - nodes[u][1]
    dz = nodes[v][2] - nodes[u][2]
    vec_uv = np.array([dx, dy, dz])
    len_uv = np.linalg.norm(vec_uv)
    
    if len_uv == 0:
        return 0.0, 1.0
        
    parent_id = nodes[u][4]
    if parent_id != -1 and parent_id in nodes:
        dx_p = nodes[u][0] - nodes[parent_id][0]
        dy_p = nodes[u][1] - nodes[parent_id][1]
        dz_p = nodes[u][2] - nodes[parent_id][2]
        vec_pu = np.array([dx_p, dy_p, dz_p])
        len_pu = np.linalg.norm(vec_pu)
        
        if len_pu == 0:
            ratio = 1.0
            theta = 0.0
        else:
            ratio = len_uv / len_pu
            cos_theta = np.dot(vec_uv, vec_pu) / (len_uv * len_pu)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
    else:
        ratio = 1.0
        theta = 0.0
        
    return theta, ratio

def extract_vectors(filepath):
    nodes, children, roots = parse_swc(filepath)
    geo_vectors = []
    inv_vectors = []
    
    for u, u_children in children.items():
        for v in u_children:
            dx = nodes[v][0] - nodes[u][0]
            dy = nodes[v][1] - nodes[u][1]
            dz = nodes[v][2] - nodes[u][2]
            dr = nodes[v][3] - nodes[u][3]
            geo_vectors.append([dx, dy, dz, dr])
            
            theta, ratio = get_inv_features(u, v, nodes)
            inv_vectors.append([theta, ratio])
            
    return geo_vectors, inv_vectors

def get_region_token(x, y, z):
    if annotation_volume is None:
        return "<REG_UNKNOWN>"
    try:
        vx = int(x / resolution)
        vy = int(y / resolution)
        vz = int(z / resolution)
        
        if (0 <= vx < annotation_volume.shape[0] and
            0 <= vy < annotation_volume.shape[1] and
            0 <= vz < annotation_volume.shape[2]):
            struct_id = annotation_volume[vx, vy, vz]
            return f"<REG_{struct_id}>"
        else:
            return "<REG_OUT_OF_BOUNDS>"
    except Exception:
        return "<REG_OUT_OF_BOUNDS>"

def process_file_worker(filepath, geo_kmeans, inv_kmeans):
    try:
        nodes, children, roots = parse_swc(filepath)
        if not roots:
            return None
            
        edge_geo_clusters = {}
        edge_inv_clusters = {}
        
        edges_list = []
        geo_coords = []
        inv_coords = []
        
        for u, u_children in children.items():
            for v in u_children:
                dx = nodes[v][0] - nodes[u][0]
                dy = nodes[v][1] - nodes[u][1]
                dz = nodes[v][2] - nodes[u][2]
                dr = nodes[v][3] - nodes[u][3]
                
                theta, ratio = get_inv_features(u, v, nodes)
                
                edges_list.append((u, v))
                geo_coords.append([dx, dy, dz, dr])
                inv_coords.append([theta, ratio])
                
        if geo_coords:
            geo_vecs = np.array(geo_coords)
            inv_vecs = np.array(inv_coords)
            
            geo_ids = geo_kmeans.predict(geo_vecs)
            inv_ids = inv_kmeans.predict(inv_vecs)
            
            for i, (u, v) in enumerate(edges_list):
                edge_geo_clusters[(u, v)] = geo_ids[i]
                edge_inv_clusters[(u, v)] = inv_ids[i]
                
        full_seq = []
        visited = set()
        last_visited_node_id = None
        
        for r in roots:
            if r in visited: continue
            
            if last_visited_node_id is not None:
                dx = nodes[r][0] - nodes[last_visited_node_id][0]
                dy = nodes[r][1] - nodes[last_visited_node_id][1]
                dz = nodes[r][2] - nodes[last_visited_node_id][2]
                jx = int(round(dx / JUMP_BIN_SIZE))
                jy = int(round(dy / JUMP_BIN_SIZE))
                jz = int(round(dz / JUMP_BIN_SIZE))
                
                jump_tok = f"<JUMP_{jx}_{jy}_{jz}>"
                full_seq.append([jump_tok, jump_tok, jump_tok])
                
            start_tok = "<START>"
            full_seq.append([start_tok, start_tok, start_tok])
            stack = [("VISIT", r)]
            
            while stack:
                action = stack.pop()
                if action[0] == "TOKEN":
                    full_seq.append([action[1], action[1], action[1]])
                elif action[0] == "TUPLE":
                    full_seq.append(action[1])
                elif action[0] == "VISIT":
                    u = action[1]
                    last_visited_node_id = u
                    
                    if u in visited:
                        continue
                    visited.add(u)
                    
                    child_ids = [c for c in children.get(u, []) if c not in visited]
                    
                    if len(child_ids) == 0:
                        continue
                        
                    actions_to_push = []
                    
                    if len(child_ids) > 1:
                        actions_to_push.append(("TOKEN", "<END_BIF>"))
                        
                    for i, v in enumerate(child_ids):
                        geo_id = edge_geo_clusters.get((u, v), 0)
                        inv_id = edge_inv_clusters.get((u, v), 0)
                        
                        reg_tok = get_region_token(nodes[v][0], nodes[v][1], nodes[v][2])
                        geo_tok = f"<GEO_{geo_id}>"
                        inv_tok = f"<INV_{inv_id}>"
                        
                        if i > 0:
                            actions_to_push.append(("TOKEN", "<POP>"))
                            
                        actions_to_push.append(("VISIT", v))
                        actions_to_push.append(("TUPLE", [inv_tok, geo_tok, reg_tok]))
                        
                    if len(child_ids) > 1:
                        actions_to_push.append(("TOKEN", "<BIF>"))
                            
                    for a in reversed(actions_to_push):
                        stack.append(a)
                        
            end_tok = "<END>"
            full_seq.append([end_tok, end_tok, end_tok])
            
        return {"file": os.path.basename(filepath), "sequence": full_seq}
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def train_vocabularies(swc_files):
    if os.path.exists(GEO_VOCAB_MODEL_PATH) and os.path.exists(INV_VOCAB_MODEL_PATH):
        print("Loading existing vocabulary models...")
        return joblib.load(GEO_VOCAB_MODEL_PATH), joblib.load(INV_VOCAB_MODEL_PATH)
        
    print(f"Phase A: Training Vocabularies on {VOCAB_TRAIN_FILES} files...")
    
    folder_map = defaultdict(list)
    for f in swc_files:
        folder = os.path.dirname(f)
        folder_map[folder].append(f)
        
    train_files = []
    while len(train_files) < VOCAB_TRAIN_FILES and folder_map:
        for folder in list(folder_map.keys()):
            if folder_map[folder]:
                train_files.append(folder_map[folder].pop())
                if len(train_files) >= VOCAB_TRAIN_FILES:
                    break
            else:
                del folder_map[folder]
                
    print(f"Selected {len(train_files)} files. Extracting vectors...")
    
    all_geo_vectors = []
    all_inv_vectors = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_vectors, f): f for f in train_files}
        for i, future in enumerate(as_completed(futures)):
            f_path = futures[future]
            try:
                g_res, i_res = future.result()
                if g_res:
                    all_geo_vectors.extend(g_res)
                    all_inv_vectors.extend(i_res)
            except Exception as e:
                print(f"DEBUG: Error extracting vectors from {f_path}: {e}")
            if (i + 1) % 10 == 0:
                print(f"  Extracted vectors from {i + 1}/{len(train_files)} files...", flush=True)
                
    def get_train_sample(vectors):
        X = np.array(vectors)
        if len(X) > MAX_SAMPLES:
            indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
            return X[indices]
        return X

    X_geo = get_train_sample(all_geo_vectors)
    X_inv = get_train_sample(all_inv_vectors)
    
    print("Training GEO K-Means...")
    geo_kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(X_geo)), random_state=42, n_init=10)
    geo_kmeans.fit(X_geo)
    
    print("Training INV K-Means...")
    inv_kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(X_inv)), random_state=42, n_init=10)
    inv_kmeans.fit(X_inv)
    
    joblib.dump(geo_kmeans, GEO_VOCAB_MODEL_PATH)
    joblib.dump(inv_kmeans, INV_VOCAB_MODEL_PATH)
    print("Models saved.")
    
    return geo_kmeans, inv_kmeans

def main():
    print(f"Finding SWC files in {SWC_DIR}...")
    swc_files = []
    for root, dirs, files in os.walk(SWC_DIR):
        for f in files:
            if f.endswith('.swc'):
                swc_files.append(os.path.join(root, f))
                
    if not swc_files:
        print("No SWC files found!")
        return
        
    if os.path.exists(GEO_VOCAB_MODEL_PATH) and os.path.exists(INV_VOCAB_MODEL_PATH):
        print("Phase A is already done. Loading existing vocabulary models...")
        geo_kmeans = joblib.load(GEO_VOCAB_MODEL_PATH)
        inv_kmeans = joblib.load(INV_VOCAB_MODEL_PATH)
    else:
        print("Phase A not done. Training vocabularies...")
        geo_kmeans, inv_kmeans = train_vocabularies(swc_files)
        
    processed_files = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"Reading already processed files from {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                idx = line.find('"file": "')
                if idx != -1:
                    start = idx + 9
                    end = line.find('"', start)
                    processed_files.add(line[start:end])
        print(f"Found {len(processed_files)} already processed files.")
        
    swc_files = [f for f in swc_files if os.path.basename(f) not in processed_files]
    
    if not swc_files:
        print("All found SWC files have already been processed! Exiting.")
        return
        
    print(f"Found {len(swc_files)} remaining files to process.")
    
    print(f"Phase B: Tokenizing {len(swc_files)} files using multiprocessing...")
        
    import concurrent.futures
    success_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
        in_flight = set()
        file_iter = iter(swc_files)
        
        with open(OUTPUT_FILE, 'a') as out_f:
            for _ in range(min(MAX_WORKERS * 2, len(swc_files))):
                try:
                    f = next(file_iter)
                    in_flight.add(executor.submit(process_file_worker, f, geo_kmeans, inv_kmeans))
                except StopIteration:
                    break
                    
            processed_so_far = 0
            while in_flight:
                done, in_flight = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        result = future.result()
                        if result:
                            out_f.write(json.dumps(result) + '\n')
                            success_count += 1
                    except Exception as e:
                        pass
                        
                    processed_so_far += 1
                    if processed_so_far % 100 == 0:
                        print(f"  Processed {processed_so_far}/{len(swc_files)} files...", flush=True)
                        
                    try:
                        f = next(file_iter)
                        in_flight.add(executor.submit(process_file_worker, f, geo_kmeans, inv_kmeans))
                    except StopIteration:
                        pass

    print(f"Done! Successfully tokenized {success_count} new neurons to {OUTPUT_FILE}.")

if __name__ == '__main__':
    main()
