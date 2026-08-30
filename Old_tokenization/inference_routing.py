"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python inference_routing.py
Input: Vesselness map and two coordinates (p1, p2)
Output: Traced optimal path connecting fragments
"""

import numpy as np
import tifffile
from skimage.graph import route_through_array
from typing import List, Tuple, Optional

def connect_fragments(vesselness_map_path: Optional[str], p1: Tuple[int, int, int], p2: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    Traces the optimal path between two points. If vesselness_map is provided, routes along the intensity ridge.
    If vesselness_map is None (arbitrary atlas space of SWCs without raw volume), routes purely based on distance.
    """
    if vesselness_map_path is not None:
        print(f"Loading vesselness map: {vesselness_map_path}")
        vesselness = tifffile.imread(vesselness_map_path)
        cost_map = 1.0 / (vesselness + 1e-6)
    else:
        print("No raw volume provided. Routing in arbitrary atlas space using uniform Euclidean costs.")
        # Create a bounding box covering p1 and p2 with some padding
        max_coord = np.maximum(p1, p2) + 20
        cost_map = np.ones(max_coord, dtype=np.float32)
    
    print(f"Routing path from {p1} to {p2}...")
    # route_through_array computes the cheapest path using Dijkstra's algorithm
    # It takes coordinates as (z, y, x)
    indices, weight = route_through_array(cost_map, p1, p2, fully_connected=True)
    
    print(f"Path found! Length: {len(indices)} voxels. Total Cost: {weight:.4f}")
    return indices

if __name__ == "__main__":
    # Test with dummy coordinates
    start_pt = (35, 64, 64)
    end_pt = (95, 64, 64)
    
    try:
        path = connect_fragments("dummy_vesselness.tif", start_pt, end_pt)
        print(f"Intensity-Guided Path Top 5: {path[:5]}")
    except FileNotFoundError:
        print("Please run skeletonize_volume.py first to generate dummy_vesselness.tif")
        
    print("\nTesting Atlas Space routing (no raw volume)...")
    path_no_vol = connect_fragments(None, start_pt, end_pt)
    print(f"Uniform Cost Path Top 5: {path_no_vol[:5]}")
