"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python detokenize_swc.py
Input: tokenized_dataset.jsonl and pre-trained K-Means vocabulary models
Output: detokenized_*.swc files reconstructed from the tokens
"""

import os
import json
import joblib
import numpy as np

with open('config.json', 'r') as f:
    config = json.load(f)

GEO_VOCAB_MODEL_PATH = config.get("GEO_VOCAB_MODEL_PATH", "kmeans_geo_vocab.pkl")
JUMP_BIN_SIZE = config.get("JUMP_BIN_SIZE", 10.0)

print(f"Loading vocabulary model from {GEO_VOCAB_MODEL_PATH}...")
kmeans = joblib.load(GEO_VOCAB_MODEL_PATH)
cluster_centers = kmeans.cluster_centers_

def detokenize_to_swc(sequence, output_file):
    """
    Converts a multi-stream sequence of tokens back into an SWC file format.
    Uses the GEO stream (index 1) for reconstruction.
    """
    nodes = {} # node_id -> (type, x, y, z, r, parent_id)
    node_counter = 1
    
    current_node_id = -1
    last_visited_node_id = -1
    bif_stack = []
    
    for seq_item in sequence:
        # We extract the token from the GEO stream (index 1)
        token = seq_item[1]
        
        if token == "<START>":
            if last_visited_node_id == -1:
                x, y, z, r = 0.0, 0.0, 0.0, 1.0
            else:
                _, last_x, last_y, last_z, last_r, _ = nodes[last_visited_node_id]
                x, y, z, r = last_x, last_y, last_z, last_r
                
            nodes[node_counter] = (2, x, y, z, r, -1)
            current_node_id = node_counter
            last_visited_node_id = node_counter
            node_counter += 1
            
        elif token.startswith("<JUMP_"):
            parts = token.strip("<>").split("_")
            jx, jy, jz = int(parts[1]), int(parts[2]), int(parts[3])
            
            if last_visited_node_id != -1:
                _, last_x, last_y, last_z, last_r, _ = nodes[last_visited_node_id]
                x = last_x + (jx * JUMP_BIN_SIZE)
                y = last_y + (jy * JUMP_BIN_SIZE)
                z = last_z + (jz * JUMP_BIN_SIZE)
                r = last_r
            else:
                x = jx * JUMP_BIN_SIZE
                y = jy * JUMP_BIN_SIZE
                z = jz * JUMP_BIN_SIZE
                r = 1.0
                
            nodes[node_counter] = (2, x, y, z, r, -1)
            last_visited_node_id = node_counter
            
        elif token == "<BIF>":
            bif_stack.append(current_node_id)
            
        elif token == "<POP>":
            if bif_stack:
                current_node_id = bif_stack[-1]
                
        elif token == "<END_BIF>":
            if bif_stack:
                bif_stack.pop()
                
        elif token.startswith("<GEO_"):
            cluster_id = int(token.strip("<>").split("_")[1])
            dx, dy, dz, dr = cluster_centers[cluster_id]
            
            _, cx, cy, cz, cr, _ = nodes[current_node_id]
            nx, ny, nz, nr = cx + dx, cy + dy, cz + dz, cr + dr
            
            nodes[node_counter] = (3, nx, ny, nz, nr, current_node_id)
            current_node_id = node_counter
            last_visited_node_id = node_counter
            node_counter += 1
            
        elif token == "<END>":
            bif_stack.clear()
            
    with open(output_file, "w") as f:
        f.write("# Detokenized SWC (Multi-Stream)\n")
        for nid, (ntype, x, y, z, r, pid) in nodes.items():
            f.write(f"{nid} {ntype} {x:.4f} {y:.4f} {z:.4f} {r:.4f} {pid}\n")
    print(f"Saved {len(nodes)} nodes to {output_file}")

if __name__ == "__main__":
    input_jsonl = config.get("OUTPUT_FILE", "tokenized_dataset.jsonl")
    if not os.path.exists(input_jsonl):
        print(f"Error: {input_jsonl} not found. Run tokenize_swc_vq.py first.")
        sys.exit(1)
        
    print(f"Reading first 3 sequences from {input_jsonl}...")
    
    try:
        count = 0
        with open(input_jsonl, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    filename = data["file"]
                    seq = data["sequence"]
                    out_name = f"detokenized_{filename}"
                    print(f"\nDetokenizing {filename} ({len(seq)} tokens)...")
                    detokenize_to_swc(seq, out_name)
                    
                    count += 1
                    if count >= 3:
                        break
    except Exception as e:
        print(f"Error: {e}")
