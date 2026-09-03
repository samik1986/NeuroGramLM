"""
Author: Samik Banerjee
Date: 2026-08-30
Usage: Main entry script to batch process SWC files through the Tokenization pipeline.
"""

import json
import logging
import os
import glob
import numpy as np
from core.pipeline import TokenizationPipeline
import concurrent.futures
import hashlib

# Setup logging for live monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def extract_fragments_from_swc(filepath):
    """
    Parses an SWC file, builds a tree, computes topological features, 
    and extracts continuous, unbranched fragments using Depth-First Traversal.
    Returns a list of fragment dicts containing points and topology.
    """
    nodes = {}
    children = {}
    intensities = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    node_id = int(parts[0])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    pid = int(parts[6])
                    
                    nodes[node_id] = np.array([x, y, z])
                    if len(parts) >= 8:
                        try:
                            intensities[node_id] = float(parts[7])
                        except ValueError:
                            intensities[node_id] = None
                    else:
                        intensities[node_id] = None

                    if pid not in children:
                        children[pid] = []
                    children[pid].append(node_id)
                    if node_id not in children:
                        children[node_id] = []
    except Exception as e:
        logging.error(f"Failed to parse {filepath}: {e}")
        return []
        
    # Find roots (nodes whose parent is -1 or not in the node list)
    roots = [nid for nid, pid_list in children.items() if nid not in nodes]
    actual_roots = []
    for fake_root in roots:
        actual_roots.extend(children[fake_root])
        
    # Topo sort / Post-order for Strahler & WL Hash
    post_order = []
    stack = list(actual_roots)
    visited = set()
    
    while stack:
        curr = stack[-1]
        if curr not in visited:
            visited.add(curr)
            for kid in children.get(curr, []):
                if kid in nodes and kid not in visited:
                    stack.append(kid)
        else:
            stack.pop()
            post_order.append(curr)
            
    seen = set()
    clean_post_order = []
    for node in post_order:
        if node not in seen:
            seen.add(node)
            clean_post_order.append(node)
            
    strahler_order = {}
    wl_hash = {}
    
    # Compute topological features
    for nid in clean_post_order:
        kids = [k for k in children.get(nid, []) if k in nodes]
        if not kids:
            strahler_order[nid] = 1
            wl_hash[nid] = int(hashlib.md5(b"leaf").hexdigest()[:8], 16)
        else:
            # Strahler Order
            child_orders = [strahler_order[k] for k in kids]
            max_order = max(child_orders)
            if child_orders.count(max_order) > 1:
                strahler_order[nid] = max_order + 1
            else:
                strahler_order[nid] = max_order
                
            # WL Hash
            child_hashes = sorted([wl_hash[k] for k in kids])
            hash_str = f"{len(kids)}_" + "_".join(map(str, child_hashes))
            wl_hash[nid] = int(hashlib.md5(hash_str.encode()).hexdigest()[:8], 16)
            
    fragments = []
    stack = []
    
    for root in actual_roots:
        if root in nodes:
            stack.append((root, []))
            
    while stack:
        current_id, current_fragment = stack.pop()
        current_fragment.append(current_id)
        kids = children.get(current_id, [])
        
        if len(kids) == 0:
            if len(current_fragment) > 1:
                fragments.append(current_fragment)
        elif len(kids) == 1:
            stack.append((kids[0], current_fragment))
        else:
            if len(current_fragment) > 1:
                fragments.append(current_fragment)
            for kid in kids:
                stack.append((kid, [current_id]))
                
    structured_fragments = []
    for frag_ids in fragments:
        frag_points = np.array([nodes[nid] for nid in frag_ids])
        topology = {
            'strahler_order': [strahler_order[nid] for nid in frag_ids],
            'wl_hash': [wl_hash[nid] for nid in frag_ids],
            'background_intensity': [intensities[nid] for nid in frag_ids]
        }
        structured_fragments.append({
            'points': frag_points,
            'topology': topology
        })
            
    return structured_fragments

def process_single_file(args):
    swc_file, config, output_dir = args
    try:
        out_filename = os.path.basename(swc_file).replace('.swc', '_tokens.json')
        out_path = os.path.join(output_dir, out_filename)
        
        if os.path.exists(out_path):
            return 1  # Already processed successfully
            
        pipeline = TokenizationPipeline(config)
        fragments = extract_fragments_from_swc(swc_file)
        if len(fragments) == 0:
            return 0
            
        all_tokens = []
        overall_quality = []
        
        for frag_data in fragments:
            result = pipeline.process_fragment(
                points=frag_data['points'], 
                topology_features=frag_data['topology'],
                available_modalities=['tortuosity', 'curvature_energy', 'inertia_tensor', 'strahler_order', 'wl_hash', 'background_intensity']
            )
            if result['success']:
                all_tokens.extend(result['tokens'])
                overall_quality.append(result['quality_score'])
                
        if len(all_tokens) > 0:
            out_filename = os.path.basename(swc_file).replace('.swc', '_tokens.json')
            out_path = os.path.join(output_dir, out_filename)
            
            with open(out_path, 'w') as f:
                json.dump({
                    'avg_quality_score': float(np.mean(overall_quality)),
                    'fragment_count': len(fragments),
                    'total_tokens': len(all_tokens),
                    'tokens': all_tokens
                }, f, indent=2)
            return 1
        return 0
    except Exception as e:
        logging.error(f"Failed processing {swc_file}: {e}")
        return 0

def main():
    logging.info("Starting Neuro_Tokenization batch pipeline...")
    
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    input_dir = config['io_paths']['input_swc_dir']
    output_dir = config['io_paths']['output_token_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    search_pattern = os.path.join(input_dir, '**', '*.swc')
    swc_files = glob.glob(search_pattern, recursive=True)
    
    if not swc_files:
        logging.warning(f"No SWC files found in {input_dir}")
        return
        
    logging.info(f"Found {len(swc_files)} SWC files. Beginning parallel processing...")
    
    success_count = 0
    args_list = [(f, config, output_dir) for f in swc_files]
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(process_single_file, args_list)):
            success_count += result
            if (i + 1) % 10 == 0 or result == 1:
                logging.info(f"Progress: {i+1}/{len(swc_files)} files processed. (Successfully tokenized: {success_count})")
            
    logging.info(f"Batch processing complete. Successfully processed {success_count}/{len(swc_files)} files.")

if __name__ == "__main__":
    main()
