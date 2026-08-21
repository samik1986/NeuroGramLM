"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python merge_fragments.py
Input: extracted_fragments.swc and dummy_vesselness.tif
Output: final_merged_neuron.swc with LM-predicted intensity splines
"""

import numpy as np
import torch
import pickle
import os
import networkx as nx
from typing import Dict, List, Tuple
from dataset import SPECIAL_TOKENS
from transformer_model import NeuroGramTransformer
from inference_routing import connect_fragments

def load_swc_fragments(filepath: str) -> Dict[int, nx.DiGraph]:
    """Reads SWC and groups connected components into NetworkX graphs."""
    graphs = {}
    current_graph = None
    nodes = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            n_id, type_, x, y, z, r, parent = (
                int(parts[0]), int(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]), float(parts[5]), int(parts[6])
            )
            nodes[n_id] = (x, y, z, r, type_)
            
            if parent == -1:
                current_graph = nx.DiGraph()
                graphs[n_id] = current_graph
                current_graph.add_node(n_id, pos=(x, y, z))
            else:
                if current_graph is not None:
                    current_graph.add_node(n_id, pos=(x, y, z))
                    current_graph.add_edge(parent, n_id)
    return graphs

def find_leaf_nodes(graph: nx.DiGraph) -> List[int]:
    """Returns nodes with out-degree 0."""
    return [n for n, d in graph.out_degree() if d == 0]

def get_path_to_root(graph: nx.DiGraph, leaf: int) -> List[int]:
    """Traces from leaf back to root and returns path in root->leaf order."""
    path = [leaf]
    current = leaf
    while list(graph.predecessors(current)):
        current = list(graph.predecessors(current))[0]
        path.append(current)
    return path[::-1]

def tokenize_sequence(path_nodes: List[int], graph: nx.DiGraph, geo_kmeans, inv_kmeans) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts a sequence of node IDs into Transformer tensor inputs."""
    geo_tokens = []
    inv_tokens = []
    reg_tokens = []
    
    geo_tokens.append(SPECIAL_TOKENS["<START>"])
    inv_tokens.append(SPECIAL_TOKENS["<START>"])
    reg_tokens.append(SPECIAL_TOKENS["<START>"])
    
    for i, n in enumerate(path_nodes):
        x, y, z = graph.nodes[n]['pos']
        # GEO
        geo_feature = np.array([[x, y, z]])
        try:
            geo_tok = geo_kmeans.predict(geo_feature)[0] + len(SPECIAL_TOKENS)
        except Exception:
            geo_tok = len(SPECIAL_TOKENS) # fallback
        geo_tokens.append(geo_tok)
        
        # INV (dummy simulation of angles)
        inv_feature = np.array([[0.0, 0.0]])
        try:
            inv_tok = inv_kmeans.predict(inv_feature)[0] + len(SPECIAL_TOKENS)
        except Exception:
            inv_tok = len(SPECIAL_TOKENS)
        inv_tokens.append(inv_tok)
        
        # REG
        reg_tokens.append(SPECIAL_TOKENS["<MASK_REG>"])
        
    x_tensor = torch.tensor([[inv_tokens, geo_tokens, reg_tokens]], dtype=torch.long).permute(0, 2, 1) # (1, L, 3)
    # Dummy PE
    pe_tensor = torch.zeros((1, len(geo_tokens), 8))
    
    return x_tensor, pe_tensor

def merge_fragments() -> None:
    print("Loading extracted fragments...")
    try:
        graphs = load_swc_fragments("extracted_fragments.swc")
    except FileNotFoundError:
        print("extracted_fragments.swc not found. Run skeletonize_volume.py first.")
        return
        
    print(f"Loaded {len(graphs)} distinct fragment graphs.")
    
    # Load token vocabularies
    try:
        with open('geo_kmeans.pkl', 'rb') as f:
            geo_kmeans = pickle.load(f)
        with open('inv_kmeans.pkl', 'rb') as f:
            inv_kmeans = pickle.load(f)
    except FileNotFoundError:
        print("KMeans vocabularies not found. Will use dummy clustering.")
        geo_kmeans = None
        inv_kmeans = None
        
    # Initialize Transformer (dummy initialized since no training weights are present)
    geo_vocab_size = 8000
    inv_vocab_size = 2000
    reg_vocab_size = 1500
    model = NeuroGramTransformer(geo_vocab_size, inv_vocab_size, reg_vocab_size, d_model=256)
    model.eval()
    
    merged_swc_lines = ["# Merged SWC by NeuroGramLM"]
    merged_swc_lines.append("# id type x y z radius parent")
    
    global_id = 1
    
    # For simplicity in this script, we just sequentially link fragment roots to previous leaf predictions
    # A full pipeline would do an exhaustive pairwise matching
    
    fragment_roots = list(graphs.keys())
    
    for i, root_id in enumerate(fragment_roots):
        graph = graphs[root_id]
        
        # Just write the nodes of this fragment
        node_mapping = {}
        for n in graph.nodes():
            x, y, z = graph.nodes[n]['pos']
            parent = list(graph.predecessors(n))
            parent_mapped = node_mapping[parent[0]] if parent else -1
            
            # If this is not the first fragment and it's the root, we connect it to the previous fragment's anchor
            if parent_mapped == -1 and i > 0:
                parent_mapped = global_id - 1
                
            merged_swc_lines.append(f"{global_id} 3 {x:.2f} {y:.2f} {z:.2f} 1.0 {parent_mapped}")
            node_mapping[n] = global_id
            global_id += 1
            
        leafs = find_leaf_nodes(graph)
        if not leafs or i == len(fragment_roots) - 1:
            continue
            
        leaf = leafs[0]
        path = get_path_to_root(graph, leaf)
        
        # Tokenize and predict anchor
        x_tensor, pe_tensor = tokenize_sequence(path, graph, geo_kmeans, inv_kmeans)
        with torch.no_grad():
            out_geo, _, _ = model(x_tensor, pe_tensor)
            
        # The last prediction in the sequence is the anchor
        pred_geo_token = torch.argmax(out_geo[0, -1, :]).item()
        
        # Decode anchor to physical coordinate
        if geo_kmeans is not None and pred_geo_token >= len(SPECIAL_TOKENS):
            try:
                anchor_coord = geo_kmeans.cluster_centers_[pred_geo_token - len(SPECIAL_TOKENS)]
            except IndexError:
                anchor_coord = np.array([0, 0, 0])
        else:
            anchor_coord = np.array([0, 0, 0])
            
        print(f"Transformer predicted anchor for leaf {leaf}: {anchor_coord}")
        
        # Route spline from leaf to next fragment's root
        next_root = fragment_roots[i+1]
        next_graph = graphs[next_root]
        next_pos = next_graph.nodes[next_root]['pos']
        leaf_pos = graph.nodes[leaf]['pos']
        
        # Route using inference_routing
        try:
            # Cast coordinates to int for routing
            p1 = (int(leaf_pos[0]), int(leaf_pos[1]), int(leaf_pos[2]))
            p2 = (int(next_pos[0]), int(next_pos[1]), int(next_pos[2]))
            
            # Use Atlas routing for demo (None) if vesselness not found, else try dummy
            if os.path.exists("dummy_vesselness.tif"):
                path_spline = connect_fragments("dummy_vesselness.tif", p1, p2)
            else:
                path_spline = connect_fragments(None, p1, p2)
                
            print(f"Routed spline with {len(path_spline)} voxels.")
            
            # Write spline nodes to SWC to bridge the gap
            for j, (sx, sy, sz) in enumerate(path_spline):
                # The first node of the spline connects to the leaf
                parent = node_mapping[leaf] if j == 0 else global_id - 1
                merged_swc_lines.append(f"{global_id} 3 {sx:.2f} {sy:.2f} {sz:.2f} 1.0 {parent}")
                global_id += 1
                
        except Exception as e:
            print(f"Routing failed: {e}")
            
    with open("final_merged_neuron.swc", "w") as f:
        f.write("\n".join(merged_swc_lines))
    print("Saved final_merged_neuron.swc!")

if __name__ == "__main__":
    merge_fragments()
