import os
import sys
sys.setrecursionlimit(100000)
import numpy as np
from sklearn.cluster import KMeans
import json
import warnings
warnings.filterwarnings('ignore')

with open('config.json', 'r') as f:
    config = json.load(f)

SWC_DIR = config.get("SWC_DIR", "SWCs/")
BATCH_SIZE = config.get("BATCH_SIZE", 50)
NUM_CLUSTERS = config.get("NUM_CLUSTERS", 128)
MAX_SAMPLES = config.get("MAX_SAMPLES_FOR_KMEANS", 100000)
JUMP_BIN_SIZE = config.get("JUMP_BIN_SIZE", 10.0)

def parse_swc(filepath):
    """Parses an SWC file into nodes and adjacency list."""
    nodes = {}
    children = {}
    
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
                
                nodes[node_id] = (x, y, z, r, n_type, parent_id)
                children[node_id] = []
                
    # Build adjacency
    roots = []
    for node_id, data in nodes.items():
        parent_id = data[5]
        # Prevent self-loops explicitly
        if parent_id == node_id:
            parent_id = -1
            
        if parent_id == -1 or parent_id not in nodes:
            roots.append(node_id)
        else:
            children[parent_id].append(node_id)
            
    return nodes, children, roots

def extract_vectors(nodes, children, roots):
    """Extracts all relative vectors (parent -> child) using Iterative DFS."""
    vectors = []
    visited = set()
    
    for r in roots:
        if r in visited: continue
        stack = [r]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            
            for v in reversed(children[u]):
                if v in visited:
                    continue
                dx = nodes[v][0] - nodes[u][0]
                dy = nodes[v][1] - nodes[u][1]
                dz = nodes[v][2] - nodes[u][2]
                dr = nodes[v][3] - nodes[u][3]
                vectors.append([dx, dy, dz, dr])
                stack.append(v)
    return vectors

def generate_sequence(nodes, children, roots, kmeans):
    """Generates the DFS token sequence iteratively to avoid recursion limits."""
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
                
                child_ids = [c for c in children[u] if c not in visited]
                
                if len(child_ids) == 0:
                    continue
                    
                actions_to_push = []
                if len(child_ids) > 1:
                    actions_to_push.append(("TOKEN", "<BIF>"))
                    
                for i, v in enumerate(child_ids):
                    dx = nodes[v][0] - nodes[u][0]
                    dy = nodes[v][1] - nodes[u][1]
                    dz = nodes[v][2] - nodes[u][2]
                    dr = nodes[v][3] - nodes[u][3]
                    
                    vec = np.array([[dx, dy, dz, dr]])
                    cluster_id = kmeans.predict(vec)[0]
                    
                    actions_to_push.append(("TOKEN", f"<GEO_{cluster_id}>"))
                    actions_to_push.append(("VISIT", v))
                    
                    # Backtrack if more children remain
                    if i < len(child_ids) - 1:
                        actions_to_push.append(("TOKEN", "<POP>"))
                        
                # Push in reverse order so they pop correctly
                for a in reversed(actions_to_push):
                    stack.append(a)
                    
        full_seq.append("<END>")
    return full_seq

def main():
    print(f"Finding SWC files in {SWC_DIR}...", flush=True)
    swc_files = []
    for root, dirs, files in os.walk(SWC_DIR):
        for f in files:
            if f.endswith('.swc'):
                swc_files.append(os.path.join(root, f))
                
    if not swc_files:
        print("No SWC files found!", flush=True)
        return
        
    swc_files = swc_files[:BATCH_SIZE]
    print(f"Processing {len(swc_files)} files to build vocabulary...", flush=True)
    
    # 1. Extract all vectors
    all_vectors = []
    parsed_data = [] # cache parsed trees
    
    for i, f in enumerate(swc_files):
        if i % 10 == 0:
            print(f"  Parsed {i}/{len(swc_files)} files...", flush=True)
        try:
            nodes, children, roots = parse_swc(f)
            parsed_data.append((f, nodes, children, roots))
            vecs = extract_vectors(nodes, children, roots)
            all_vectors.extend(vecs)
        except Exception as e:
            print(f"Error parsing {f}: {e}", flush=True)
            
    print(f"Extracted {len(all_vectors)} directional vectors. Training K-Means (K={NUM_CLUSTERS})...", flush=True)
    
    # 2. Train K-Means (Optimized with Subsampling)
    X = np.array(all_vectors)
    if len(X) > MAX_SAMPLES:
        print(f"Subsampling {len(X)} down to {MAX_SAMPLES} for fast VQ training...", flush=True)
        indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X_train = X[indices]
    else:
        X_train = X
        
    kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(X_train)), random_state=42, n_init=10)
    kmeans.fit(X_train)
    print("K-Means training complete.", flush=True)
    
    # 3. Generate Token Sequences
    print("Generating SMILES-like sequences...", flush=True)
    results = []
    for f, nodes, children, roots in parsed_data:
        seq = generate_sequence(nodes, children, roots, kmeans)
        results.append({
            'file': os.path.basename(f),
            'sequence': " ".join(seq),
            'num_nodes': len(nodes)
        })
        
    # Save results
    output_file = 'tokenized_batch.json'
    with open(output_file, 'w') as out_f:
        json.dump(results, out_f, indent=2)
        
    print(f"\nSaved {len(results)} tokenized sequences to {output_file}.", flush=True)
    print("\nSample sequence (first 50 tokens):", flush=True)
    if results:
        sample_seq = results[0]['sequence'].split()
        print(" ".join(sample_seq[:50]) + ("..." if len(sample_seq) > 50 else ""), flush=True)

if __name__ == "__main__":
    main()
