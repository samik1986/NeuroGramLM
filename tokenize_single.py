"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python tokenize_single.py <path_to_swc> [output_jsonl]
Input: A single .swc file path, pre-trained K-Means models, and annotation_25.nrrd
Output: A tokenized sequence jsonl file for the specific neuron
"""

import os
import sys
import numpy as np
import json
import joblib
import nrrd
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

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

def get_region_token(x, y, z, annotation_volume, resolution=25):
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

def tokenize_single(filepath, output_filepath=None):
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    GEO_VOCAB_MODEL = config.get("GEO_VOCAB_MODEL_PATH", "kmeans_geo_vocab.pkl")
    INV_VOCAB_MODEL = config.get("INV_VOCAB_MODEL_PATH", "kmeans_inv_vocab.pkl")
    JUMP_BIN_SIZE = config.get("JUMP_BIN_SIZE", 10.0)
    
    try:
        geo_kmeans = joblib.load(GEO_VOCAB_MODEL)
        inv_kmeans = joblib.load(INV_VOCAB_MODEL)
    except Exception as e:
        print(f"Error loading vocabularies: {e}")
        return None
        
    try:
        annotation_volume, _ = nrrd.read('annotation_25.nrrd')
    except Exception as e:
        print(f"Error loading annotation_25.nrrd: {e}")
        annotation_volume = None

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
        geo_ids = geo_kmeans.predict(np.array(geo_coords))
        inv_ids = inv_kmeans.predict(np.array(inv_coords))
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
                if u in visited: continue
                visited.add(u)
                
                child_ids = [c for c in children.get(u, []) if c not in visited]
                if not child_ids: continue
                
                actions_to_push = []
                if len(child_ids) > 1:
                    actions_to_push.append(("TOKEN", "<END_BIF>"))
                    
                for i, v in enumerate(child_ids):
                    geo_id = edge_geo_clusters.get((u, v), 0)
                    inv_id = edge_inv_clusters.get((u, v), 0)
                    reg_tok = get_region_token(nodes[v][0], nodes[v][1], nodes[v][2], annotation_volume)
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
        
    result = {"file": os.path.basename(filepath), "sequence": full_seq}
    
    if output_filepath:
        with open(output_filepath, 'w') as f:
            f.write(json.dumps(result) + '\n')
        print(f"Tokenized {filepath} -> {output_filepath}")
        
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tokenize_single.py <path_to_swc> [output_jsonl]")
        sys.exit(1)
        
    swc_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"tokenized_{os.path.basename(swc_path)}.jsonl"
    tokenize_single(swc_path, out_path)
