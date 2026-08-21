import os
import sys
import numpy as np
import joblib
import json

SWC_DIR = "SWCs_test"
REL_LOC_VOCAB_PATH = "kmeans_rel_loc.pkl"
OUTPUT_DIR = "detokenized_fragments"
INPUT_FILE = "tokenized_fragments.jsonl"

def detokenize_file(data):
    filename = data["file"]
    fragments = data["fragments"]
    
    nodes = {}
    
    loc_kmeans = joblib.load(REL_LOC_VOCAB_PATH)
    loc_centers = loc_kmeans.cluster_centers_
    
    # We will generate sequential IDs for nodes to build the SWC,
    # but we need to map the original start/end/parent nodes to our new IDs.
    # Since we can just use the original IDs to preserve them, it's easier!
    
    # Fragments might not be ordered. We can just process them, but to 
    # place them in space, we need the parent to be processed first to get the initial coordinate.
    
    # Build a dependency graph of fragments
    frag_by_start = {f["start_node"]: f for f in fragments}
    
    placed_coords = {}
    
    # Find roots (fragments with parent -1 or parent not in frag_by_start)
    # Actually, a parent could be the end_node or any node of another fragment.
    # We need to map original node ID -> (x,y,z,r).
    
    def process_frag(frag):
        start_id = frag["start_node"]
        if start_id in placed_coords:
            return # already processed
            
        parent_id = frag["parent_node"]
        
        # Determine starting coordinate
        if parent_id in placed_coords:
            x, y, z = placed_coords[parent_id]
        else:
            x, y, z = 0.0, 0.0, 0.0
            
        r = 1.0 # default radius
        
        # Add the first node of the fragment
        nodes[start_id] = (x, y, z, r, parent_id)
        placed_coords[start_id] = (x, y, z)
        
        curr_x, curr_y, curr_z = x, y, z
        
        # We need to generate IDs for the intermediate nodes.
        # Since we don't have their original IDs, we generate new ones starting from max(original) + 1
        # But wait, we can just assign new IDs sequentially, except for start and end which must be preserved?
        # Actually, let's just generate all new IDs and keep a mapping from original -> new.
        pass
        
    # Let's simplify: just use entirely new IDs for everything.
    new_nodes = {} # new_id -> (type, x, y, z, r, parent_id)
    orig_to_new = {} # original_id -> new_id
    current_new_id = 1
    
    # Sort fragments: roots first, then those whose parents are processed
    pending = list(fragments)
    
    while pending:
        progress = False
        for frag in pending[:]:
            p_id = frag["parent_node"]
            
            if p_id == -1 or p_id in orig_to_new:
                # We can process this fragment
                pending.remove(frag)
                progress = True
                
                seq = frag["sequence"]
                # seq = [START_TOK, [LOC, DIR, MORPH, REG], ..., END_TOK]
                start_tok = seq[0]
                end_tok = seq[-1]
                steps = seq[1:-1]
                
                if p_id in orig_to_new:
                    # start at parent's coordinate
                    parent_new_id = orig_to_new[p_id]
                    px, py, pz = new_nodes[parent_new_id][1:4]
                    pr = new_nodes[parent_new_id][4]
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                    pr = 1.0
                    parent_new_id = -1
                    
                curr_x, curr_y, curr_z = px, py, pz
                curr_r = pr
                prev_new_id = parent_new_id
                
                # First node of the fragment (if it's a SOMA or FRAG, it's a new node. If BIF, it might be the same as parent? No, in SWC, child is a new node connected to parent)
                first_node_new_id = current_new_id
                current_new_id += 1
                orig_to_new[frag["start_node"]] = first_node_new_id
                
                node_type = 1 if start_tok == "<START_SOMA>" else 3
                new_nodes[first_node_new_id] = (node_type, curr_x, curr_y, curr_z, curr_r, prev_new_id)
                prev_new_id = first_node_new_id
                
                for step in steps:
                    # step is [LOC, DIR, MORPH, REG]
                    loc_tok = step[0]
                    # parse loc_tok: <REL_LOC_42>
                    if loc_tok.startswith("<REL_LOC_") and loc_tok.endswith(">"):
                        loc_id = int(loc_tok[9:-1])
                        dx, dy, dz = loc_centers[loc_id]
                        curr_x += dx
                        curr_y += dy
                        curr_z += dz
                        
                    # We just use constant radius for now since we didn't perfectly track r reconstruction
                    curr_r = 1.0
                    
                    next_node_id = current_new_id
                    current_new_id += 1
                    
                    new_nodes[next_node_id] = (3, curr_x, curr_y, curr_z, curr_r, prev_new_id)
                    prev_new_id = next_node_id
                    
                orig_to_new[frag["end_node"]] = prev_new_id
                
        if not progress:
            # If there are pending fragments but no progress, they are disconnected and their parent is missing!
            # Treat them as roots.
            for frag in pending:
                frag["parent_node"] = -1
            
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, 'w') as f:
        f.write("# Detokenized from fragment tokens\n")
        for n_id, data in sorted(new_nodes.items()):
            ntype, x, y, z, r, pid = data
            f.write(f"{n_id} {ntype} {x:.4f} {y:.4f} {z:.4f} {r:.2f} {pid}\n")
            
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found!")
        return
        
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            print(f"Detokenizing {data['file']}...")
            detokenize_file(data)
            
    print("Done!")

if __name__ == '__main__':
    main()
