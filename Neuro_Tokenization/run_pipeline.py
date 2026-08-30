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

# Setup logging for live monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def extract_fragments_from_swc(filepath):
    """
    Parses an SWC file, builds a tree, and extracts continuous, unbranched 
    fragments using Depth-First Traversal.
    Returns a list of fragments, where each fragment is a numpy array of 3D points.
    """
    nodes = {}
    children = {}
    
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
                    if pid not in children:
                        children[pid] = []
                    children[pid].append(node_id)
                    if node_id not in children:
                        children[node_id] = []
    except Exception as e:
        logging.error(f"Failed to parse {filepath}: {e}")
        return []
        
    # Find roots (nodes whose parent is -1 or not in the node list)
    roots = [nid for nid in nodes.keys() if all(pid != nid for pid in children.keys() if pid in nodes)]
    # A simpler way to find roots in SWC: usually pid is -1
    roots = [nid for nid, pid_list in children.items() if nid not in nodes]
    actual_roots = []
    for fake_root in roots:
        actual_roots.extend(children[fake_root])
        
    fragments = []
    stack = []
    
    for root in actual_roots:
        if root in nodes:
            stack.append((root, []))
            
    while stack:
        current_id, current_fragment = stack.pop()
        current_fragment.append(nodes[current_id])
        kids = children.get(current_id, [])
        
        if len(kids) == 0:
            if len(current_fragment) > 1:
                fragments.append(np.array(current_fragment))
        elif len(kids) == 1:
            stack.append((kids[0], current_fragment))
        else:
            if len(current_fragment) > 1:
                fragments.append(np.array(current_fragment))
            for kid in kids:
                stack.append((kid, [nodes[current_id]]))
            
    return fragments

def main():
    logging.info("Starting Neuro_Tokenization batch pipeline...")
    
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    input_dir = config['io_paths']['input_directory']
    output_dir = config['io_paths']['output_directory']
    
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = TokenizationPipeline(config)
    
    # Find all SWC files recursively
    search_pattern = os.path.join(input_dir, '**', '*.swc')
    swc_files = glob.glob(search_pattern, recursive=True)
    
    if not swc_files:
        logging.warning(f"No SWC files found in {input_dir}")
        return
        
    logging.info(f"Found {len(swc_files)} SWC files. Beginning processing...")
    
    success_count = 0
    
    # Process files (just limit to 10 for demonstration/testing if there are thousands)
    for i, swc_file in enumerate(swc_files):
        logging.info(f"Processing ({i+1}/{len(swc_files)}): {swc_file}")
        
        fragments = extract_fragments_from_swc(swc_file)
        if len(fragments) == 0:
            logging.warning(f"Skipping empty or invalid file: {swc_file}")
            continue
            
        all_tokens = []
        overall_quality = []
        
        for frag in fragments:
            result = pipeline.process_fragment(frag)
            if result['success']:
                all_tokens.extend(result['tokens'])
                overall_quality.append(result['quality_score'])
                
        if len(all_tokens) > 0:
            # Save the tokens
            out_filename = os.path.basename(swc_file).replace('.swc', '_tokens.json')
            out_path = os.path.join(output_dir, out_filename)
            
            with open(out_path, 'w') as f:
                json.dump({
                    'avg_quality_score': float(np.mean(overall_quality)),
                    'fragment_count': len(fragments),
                    'total_tokens': len(all_tokens),
                    'tokens': all_tokens
                }, f, indent=2)
                
            logging.info(f"Saved {len(all_tokens)} tokens from {len(fragments)} fragments to {out_path}")
            success_count += 1
        else:
            logging.error(f"Pipeline failed for all fragments in {swc_file}")
            
    logging.info(f"Batch processing complete. Successfully processed {success_count}/{len(swc_files)} files.")

if __name__ == "__main__":
    main()
