"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python compare_swc.py
Input: Original .swc files in SWCs/ and reconstructed detokenized_*.swc files
Output: Terminal printout of topology comparison metrics (Node count, Path length, BBox)
"""

import os
import glob
import sys
import numpy as np

def parse_swc_stats(filepath):
    nodes = {}
    total_length = 0.0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                nid = int(parts[0])
                x, y, z = map(float, parts[2:5])
                pid = int(parts[6])
                nodes[nid] = np.array([x, y, z])
                
                if pid != -1 and pid in nodes:
                    dist = np.linalg.norm(nodes[nid] - nodes[pid])
                    total_length += dist
                    
    if not nodes:
        return None
        
    coords = np.array(list(nodes.values()))
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    
    return {
        "num_nodes": len(nodes),
        "total_length": total_length,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max
    }

def find_original_swc(filename, search_dir="SWCs_test/"):
    for root, _, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def compare_swcs():
    detokenized_files = glob.glob("detokenized_*.swc")
    if not detokenized_files:
        print("No detokenized SWC files found. Run detokenize_swc.py first.")
        return
        
    for detok_file in detokenized_files:
        orig_filename = detok_file.replace("detokenized_", "")
        orig_file = find_original_swc(orig_filename)
        
        print(f"\n{'='*50}")
        print(f"Comparing: {orig_filename}")
        print(f"{'='*50}")
        
        if not orig_file:
            print(f"ERROR: Could not find original file {orig_filename} in SWCs/")
            continue
            
        orig_stats = parse_swc_stats(orig_file)
        detok_stats = parse_swc_stats(detok_file)
        
        if not orig_stats or not detok_stats:
            print("Error parsing one of the files.")
            continue
            
        print(f"{'Metric':<20} | {'Original':<20} | {'Detokenized (VQ)':<20}")
        print("-" * 65)
        print(f"{'Nodes':<20} | {orig_stats['num_nodes']:<20} | {detok_stats['num_nodes']:<20}")
        print(f"{'Total Length (um)':<20} | {orig_stats['total_length']:<20.2f} | {detok_stats['total_length']:<20.2f}")
        
        orig_bbox = f"{orig_stats['bbox_max'] - orig_stats['bbox_min']}"
        detok_bbox = f"{detok_stats['bbox_max'] - detok_stats['bbox_min']}"
        print(f"{'BBox Extent (X,Y,Z)':<20} | {orig_bbox:<20} | {detok_bbox:<20}")
        
        diff_len = abs(orig_stats['total_length'] - detok_stats['total_length'])
        pct_len = (diff_len / orig_stats['total_length']) * 100 if orig_stats['total_length'] > 0 else 0
        print(f"\nTotal length difference: {diff_len:.2f} um ({pct_len:.2f}%)")

if __name__ == "__main__":
    compare_swcs()
