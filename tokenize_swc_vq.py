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
NUM_CLUSTERS = config.get("NUM_CLUSTERS", 128)
JUMP_BIN_SIZE = config.get("JUMP_BIN_SIZE", 10.0)
VOCAB_MODEL_PATH = config.get("VOCAB_MODEL_PATH", "kmeans_vocab.pkl")
VOCAB_TRAIN_FILES = config.get("VOCAB_TRAIN_FILES", 1000)
MAX_SAMPLES = config.get("MAX_SAMPLES_FOR_KMEANS", 100000)
MAX_WORKERS = config.get("MAX_WORKERS", 16)
OUTPUT_FILE = config.get("OUTPUT_FILE", "tokenized_dataset.jsonl")

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
                
                nodes[node_id] = (x, y, z, r)
                if parent_id != -1 and parent_id in nodes:
                    children[parent_id].append(node_id)
                    
    roots = [n for n in nodes.keys() if all(n not in v for v in children.values())]
    return nodes, dict(children), roots

def extract_vectors(filepath):
    nodes, children, roots = parse_swc(filepath)
    vectors = []
    for u, u_children in children.items():
        for v in u_children:
            dx = nodes[v][0] - nodes[u][0]
            dy = nodes[v][1] - nodes[u][1]
            dz = nodes[v][2] - nodes[u][2]
            dr = nodes[v][3] - nodes[u][3]
            vectors.append([dx, dy, dz, dr])
    return vectors

def process_file_worker(filepath, kmeans_model):
    """Worker function for multiprocessing."""
    try:
        nodes, children, roots = parse_swc(filepath)
        if not roots:
            return None
            
        # 1. Vectorized Prediction
        edge_clusters = {}
        edges_list = []
        edges_coords = []
        
        for u, u_children in children.items():
            for v in u_children:
                dx = nodes[v][0] - nodes[u][0]
                dy = nodes[v][1] - nodes[u][1]
                dz = nodes[v][2] - nodes[u][2]
                dr = nodes[v][3] - nodes[u][3]
                edges_list.append((u, v))
                edges_coords.append([dx, dy, dz, dr])
                
        if edges_coords:
            vecs = np.array(edges_coords)
            cluster_ids = kmeans_model.predict(vecs)
            for i, (u, v) in enumerate(edges_list):
                edge_clusters[(u, v)] = cluster_ids[i]
                
        # 2. Iterative DFS Sequence Generation
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
                full_seq.append(f"<JUMP_{jx}_{jy}_{jz}>")
                
            full_seq.append("<START>")
            stack = [("VISIT", r)]
            
            while stack:
                action = stack.pop()
                if action[0] == "TOKEN":
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
                        actions_to_push.append(("TOKEN", "<BIF>"))
                        
                    for i, v in enumerate(child_ids):
                        # Fast lookup instead of predict
                        cluster_id = edge_clusters.get((u, v), 0)
                        actions_to_push.append(("TOKEN", f"<GEO_{cluster_id}>"))
                        actions_to_push.append(("VISIT", v))
                        
                        if i < len(child_ids) - 1:
                            actions_to_push.append(("TOKEN", "<POP>"))
                            
                    for a in reversed(actions_to_push):
                        stack.append(a)
                        
            full_seq.append("<END>")
            
        return {"file": os.path.basename(filepath), "sequence": full_seq}
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def train_vocabulary(swc_files):
    if os.path.exists(VOCAB_MODEL_PATH):
        print(f"Loading existing vocabulary model from {VOCAB_MODEL_PATH}...")
        return joblib.load(VOCAB_MODEL_PATH)
        
    print(f"Phase A: Training Vocabulary on {VOCAB_TRAIN_FILES} files...")
    # Stratified random sampling across folders (by clustering directories)
    folder_map = defaultdict(list)
    for f in swc_files:
        folder = os.path.dirname(f)
        folder_map[folder].append(f)
        
    train_files = []
    # Take evenly from each folder to reach VOCAB_TRAIN_FILES
    while len(train_files) < VOCAB_TRAIN_FILES and folder_map:
        for folder in list(folder_map.keys()):
            if folder_map[folder]:
                train_files.append(folder_map[folder].pop())
                if len(train_files) >= VOCAB_TRAIN_FILES:
                    break
            else:
                del folder_map[folder]
                
    print(f"Selected {len(train_files)} files. Extracting vectors...")
    all_vectors = []
    for f in train_files:
        all_vectors.extend(extract_vectors(f))
        
    X = np.array(all_vectors)
    if len(X) > MAX_SAMPLES:
        print(f"Subsampling {len(X)} down to {MAX_SAMPLES} for fast VQ training...")
        indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X_train = X[indices]
    else:
        X_train = X
        
    print(f"Training K-Means (K={NUM_CLUSTERS})...")
    kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(X_train)), random_state=42, n_init=10)
    kmeans.fit(X_train)
    
    joblib.dump(kmeans, VOCAB_MODEL_PATH)
    print(f"Vocabulary model saved to {VOCAB_MODEL_PATH}.")
    return kmeans

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
        
    kmeans = train_vocabulary(swc_files)
    
    print(f"Phase B: Tokenizing {len(swc_files)} files using multiprocessing (Workers: {MAX_WORKERS})...")
    
    # Empty the jsonl file if it exists
    with open(OUTPUT_FILE, 'w') as f:
        pass
        
    success_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        future_to_file = {executor.submit(process_file_worker, f, kmeans): f for f in swc_files}
        
        with open(OUTPUT_FILE, 'a') as out_f:
            for i, future in enumerate(as_completed(future_to_file)):
                result = future.result()
                if result:
                    out_f.write(json.dumps(result) + '\n')
                    success_count += 1
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(swc_files)} files...", flush=True)

    print(f"Done! Successfully tokenized {success_count} neurons to {OUTPUT_FILE}.")

if __name__ == '__main__':
    main()
