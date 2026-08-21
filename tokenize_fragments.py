import os
import sys
import numpy as np
from sklearn.cluster import KMeans
import json
import warnings
import joblib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

with open('config.json', 'r') as f:
    config = json.load(f)

SWC_DIR = config.get("SWC_DIR", "SWCs_test/")
NUM_CLUSTERS = config.get("NUM_CLUSTERS", 512)
REL_LOC_VOCAB_PATH = config.get("REL_LOC_VOCAB_PATH", "kmeans_rel_loc.pkl")
DIR_VOCAB_PATH = config.get("DIR_VOCAB_PATH", "kmeans_dir.pkl")
MORPH_VOCAB_PATH = config.get("MORPH_VOCAB_PATH", "kmeans_morph.pkl")
VOCAB_TRAIN_FILES = config.get("VOCAB_TRAIN_FILES", 1000)
MAX_SAMPLES = config.get("MAX_SAMPLES_FOR_KMEANS", 100000)
MAX_WORKERS = config.get("MAX_WORKERS", 16)
OUTPUT_FILE = config.get("OUTPUT_FILE", "tokenized_fragments.jsonl")

# Global atlas for workers
annotation_volume = None
resolution = 25

def init_worker() -> None:
    global annotation_volume, resolution
    try:
        import nrrd
        if os.path.exists('annotation_25.nrrd'):
            data, header = nrrd.read('annotation_25.nrrd')
            annotation_volume = data
    except Exception as e:
        print(f"Worker atlas init failed: {e}")
        annotation_volume = None

def parse_swc(filepath: str) -> Tuple[Dict[int, Tuple[float, float, float, float, int]], Dict[int, List[int]], List[int]]:
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
                x, y, z, r = map(float, parts[2:6])
                parent_id = int(parts[6])
                
                nodes[node_id] = (x, y, z, r, parent_id)
                if parent_id != -1 and parent_id in nodes:
                    children[parent_id].append(node_id)
                    
    all_children = set(child for child_list in children.values() for child in child_list)
    roots = [n for n in nodes.keys() if n not in all_children]
    return nodes, dict(children), roots

def extract_fragments(nodes: Dict[int, Tuple[float, float, float, float, int]], children: Dict[int, List[int]], roots: List[int]) -> List[Dict]:
    start_nodes = list(roots)
    for u, ch in children.items():
        if len(ch) > 1:
            start_nodes.extend(ch)
            
    fragments = []
    for s in start_nodes:
        frag = [s]
        curr = s
        visited_in_frag = {s}
        while True:
            ch = children.get(curr, [])
            if len(ch) == 0:
                break
            elif len(ch) > 1:
                break
            else:
                curr = ch[0]
                if curr in visited_in_frag:
                    break
                visited_in_frag.add(curr)
                frag.append(curr)
                
        p_id = nodes[s][4]
        if p_id == -1 or p_id not in nodes:
            start_type = "<START_SOMA>"
        else:
            start_type = "<START_BIF>"
            
        end_ch = children.get(frag[-1], [])
        if len(end_ch) == 0:
            end_type = "<END_LEAF>"
        else:
            end_type = "<END_BIF>"
            
        fragments.append({
            "nodes": frag,
            "start_type": start_type,
            "end_type": end_type,
            "parent_node": p_id
        })
    return fragments

def get_region_token(x: float, y: float, z: float) -> str:
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

def extract_fragment_vectors(nodes: Dict, frag_nodes: List[int]) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[str]]:
    rel_locs = []
    dirs = []
    morphs = []
    regs = []
    
    if len(frag_nodes) < 2:
        return rel_locs, dirs, morphs, regs
        
    lengths = []
    for i in range(len(frag_nodes) - 1):
        u, v = frag_nodes[i], frag_nodes[i+1]
        dx, dy, dz = nodes[v][0] - nodes[u][0], nodes[v][1] - nodes[u][1], nodes[v][2] - nodes[u][2]
        lengths.append(np.sqrt(dx**2 + dy**2 + dz**2))
        
    med_len = np.median(lengths) if lengths else 1.0
    if med_len == 0:
        med_len = 1.0
        
    prev_dir = None
    prev_len = None
    
    for i in range(len(frag_nodes) - 1):
        u, v = frag_nodes[i], frag_nodes[i+1]
        dx, dy, dz = nodes[v][0] - nodes[u][0], nodes[v][1] - nodes[u][1], nodes[v][2] - nodes[u][2]
        curr_len = np.sqrt(dx**2 + dy**2 + dz**2)
        
        rel_loc = [dx, dy, dz]
        
        if curr_len > 0:
            curr_dir = [dx / curr_len, dy / curr_len, dz / curr_len]
        else:
            curr_dir = [0.0, 0.0, 0.0]
            
        rad_ratio = nodes[v][3] / nodes[u][3] if nodes[u][3] > 0 else 1.0
        
        if prev_dir is not None and prev_len is not None and prev_len > 0:
            len_ratio = curr_len / prev_len
            cos_theta = np.dot(curr_dir, prev_dir)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
        else:
            len_ratio = 1.0
            cos_theta = 1.0
            
        morph = [cos_theta, rad_ratio, len_ratio]
        reg = get_region_token(nodes[u][0], nodes[u][1], nodes[u][2])
        
        rel_locs.append(rel_loc)
        dirs.append(curr_dir)
        morphs.append(morph)
        regs.append(reg)
        
        prev_dir = curr_dir
        prev_len = curr_len
        
    return rel_locs, dirs, morphs, regs

def extract_all_vectors(filepath: str) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    nodes, children, roots = parse_swc(filepath)
    fragments = extract_fragments(nodes, children, roots)
    
    all_rel_locs = []
    all_dirs = []
    all_morphs = []
    
    for frag in fragments:
        rel_locs, dirs, morphs, _ = extract_fragment_vectors(nodes, frag["nodes"])
        all_rel_locs.extend(rel_locs)
        all_dirs.extend(dirs)
        all_morphs.extend(morphs)
        
    return all_rel_locs, all_dirs, all_morphs

def train_vocabularies(swc_files: List[str]) -> Tuple[KMeans, KMeans, KMeans]:
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
    
    total_rel_locs = []
    total_dirs = []
    total_morphs = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_all_vectors, f): f for f in train_files}
        for i, future in enumerate(as_completed(futures)):
            f_path = futures[future]
            try:
                r_res, d_res, m_res = future.result()
                total_rel_locs.extend(r_res)
                total_dirs.extend(d_res)
                total_morphs.extend(m_res)
            except Exception as e:
                print(f"Error extracting vectors from {f_path}: {e}")
            if (i + 1) % 10 == 0:
                print(f"  Extracted vectors from {i + 1}/{len(train_files)} files...", flush=True)
                
    def get_train_sample(vectors):
        X = np.array(vectors)
        if len(X) > MAX_SAMPLES:
            indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
            return X[indices]
        return X

    X_loc = get_train_sample(total_rel_locs)
    X_dir = get_train_sample(total_dirs)
    X_morph = get_train_sample(total_morphs)
    
    print("Training REL_LOC K-Means...")
    loc_kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(X_loc)), random_state=42, n_init=10)
    loc_kmeans.fit(X_loc)
    
    print("Training DIR K-Means...")
    dir_kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(X_dir)), random_state=42, n_init=10)
    dir_kmeans.fit(X_dir)

    print("Training MORPH K-Means...")
    morph_kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(X_morph)), random_state=42, n_init=10)
    morph_kmeans.fit(X_morph)
    
    joblib.dump(loc_kmeans, REL_LOC_VOCAB_PATH)
    joblib.dump(dir_kmeans, DIR_VOCAB_PATH)
    joblib.dump(morph_kmeans, MORPH_VOCAB_PATH)
    print("Models saved.")
    
    return loc_kmeans, dir_kmeans, morph_kmeans

def process_file_worker(filepath: str, loc_kmeans: KMeans, dir_kmeans: KMeans, morph_kmeans: KMeans) -> Optional[Dict[str, Any]]:
    try:
        nodes, children, roots = parse_swc(filepath)
        fragments = extract_fragments(nodes, children, roots)
        
        sequence_data = []
        
        for frag in fragments:
            rel_locs, dirs, morphs, regs = extract_fragment_vectors(nodes, frag["nodes"])
            
            if not rel_locs:
                continue
                
            loc_ids = loc_kmeans.predict(np.array(rel_locs))
            dir_ids = dir_kmeans.predict(np.array(dirs))
            morph_ids = morph_kmeans.predict(np.array(morphs))
            
            frag_seq = [frag["start_type"]]
            for i in range(len(rel_locs)):
                tup = [f"<REL_LOC_{loc_ids[i]}>", f"<DIR_{dir_ids[i]}>", f"<MORPH_{morph_ids[i]}>", regs[i]]
                frag_seq.append(tup)
            frag_seq.append(frag["end_type"])
            
            sequence_data.append({
                "start_node": frag["nodes"][0],
                "end_node": frag["nodes"][-1],
                "parent_node": frag["parent_node"],
                "sequence": frag_seq
            })
            
        return {"file": os.path.basename(filepath), "fragments": sequence_data}
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main() -> None:
    print(f"Finding SWC files in {SWC_DIR}...")
    swc_files = []
    for root, dirs, files in os.walk(SWC_DIR):
        for f in files:
            if f.endswith('.swc'):
                swc_files.append(os.path.join(root, f))
                
    if not swc_files:
        print("No SWC files found!")
        return
        
    if os.path.exists(REL_LOC_VOCAB_PATH) and os.path.exists(DIR_VOCAB_PATH) and os.path.exists(MORPH_VOCAB_PATH):
        print("Loading existing vocabulary models...")
        loc_kmeans = joblib.load(REL_LOC_VOCAB_PATH)
        dir_kmeans = joblib.load(DIR_VOCAB_PATH)
        morph_kmeans = joblib.load(MORPH_VOCAB_PATH)
    else:
        print("Training vocabularies...")
        loc_kmeans, dir_kmeans, morph_kmeans = train_vocabularies(swc_files)
        
    processed_files = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                idx = line.find('"file": "')
                if idx != -1:
                    start = idx + 9
                    end = line.find('"', start)
                    processed_files.add(line[start:end])
                    
    swc_files = [f for f in swc_files if os.path.basename(f) not in processed_files]
    
    if not swc_files:
        print("All found SWC files have already been processed! Exiting.")
        return
        
    print(f"Phase B: Tokenizing {len(swc_files)} files using multiprocessing...")
        
    success_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
        in_flight = set()
        file_iter = iter(swc_files)
        
        with open(OUTPUT_FILE, 'a') as out_f:
            for _ in range(min(MAX_WORKERS * 2, len(swc_files))):
                try:
                    f = next(file_iter)
                    in_flight.add(executor.submit(process_file_worker, f, loc_kmeans, dir_kmeans, morph_kmeans))
                except StopIteration:
                    break
                    
            processed_so_far = 0
            while in_flight:
                import concurrent.futures
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
                        in_flight.add(executor.submit(process_file_worker, f, loc_kmeans, dir_kmeans, morph_kmeans))
                    except StopIteration:
                        pass

    print(f"Done! Successfully tokenized {success_count} new neurons to {OUTPUT_FILE}.")

if __name__ == '__main__':
    main()
