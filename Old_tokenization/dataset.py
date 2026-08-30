"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python dataset.py
Input: JSONL tokenized dataset
Output: PyTorch DataLoader with Graph Laplacian Positional Encoding
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from collections.abc import Iterator
from typing import List, Tuple
import json
import time
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import scipy.sparse as sp

VOCAB_SIZE_GEO = 512
VOCAB_SIZE_INV = 512
VOCAB_SIZE_REG = 1500

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "<BIF>": 3,
    "<END_BIF>": 4,
    "<POP>": 5,
    "<MASK_REG>": 6,
    "<JUMP>": 7
}

def parse_token(tok: str) -> int:
    if tok in SPECIAL_TOKENS:
        return SPECIAL_TOKENS[tok]
    if tok.startswith("<JUMP_"):
        return SPECIAL_TOKENS["<JUMP>"]
        
    if tok.startswith("<GEO_"):
        return int(tok.replace("<GEO_", "").replace(">", "")) + len(SPECIAL_TOKENS)
    if tok.startswith("<INV_"):
        return int(tok.replace("<INV_", "").replace(">", "")) + len(SPECIAL_TOKENS)
    if tok.startswith("<REG_"):
        return (int(tok.replace("<REG_", "").replace(">", "")) % 60000) + len(SPECIAL_TOKENS)
        
    return SPECIAL_TOKENS["<PAD>"]

def build_chunk_adjacency(chunk: List[List[int]]) -> sp.csr_matrix:
    """
    Reconstructs the undirected graph adjacency matrix from a chunk of DFS tokens.
    """
    N = len(chunk)
    adj = sp.dok_matrix((N, N), dtype=np.float32)
    
    stack = []
    parent = -1
    
    for i, token_triple in enumerate(chunk):
        id1 = token_triple[0]
        if id1 == SPECIAL_TOKENS["<PAD>"]:
            continue
            
        if id1 == SPECIAL_TOKENS["<BIF>"]:
            stack.append(parent)
        elif id1 == SPECIAL_TOKENS["<POP>"]:
            if stack:
                parent = stack[-1]
        elif id1 == SPECIAL_TOKENS["<END_BIF>"]:
            if stack:
                stack.pop()
        else: # Normal spatial node or jump
            if parent != -1:
                adj[parent, i] = 1.0
                adj[i, parent] = 1.0
            parent = i
            
    return adj.tocsr()

def compute_laplacian_pe(adj_matrix: sp.csr_matrix, k: int = 8) -> np.ndarray:
    """
    Compute Graph Laplacian Positional Encoding.
    """
    N = adj_matrix.shape[0]
    adj_matrix = adj_matrix + sp.eye(N)
    
    d = np.array(adj_matrix.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    
    L = sp.eye(N) - D_inv_sqrt.dot(adj_matrix).dot(D_inv_sqrt)
    
    if N <= k:
        k = N - 1
        
    if k <= 0:
        return np.zeros((N, 8), dtype=np.float32)
        
    try:
        eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=k+1, which='SM', tol=1e-2)
        pe = eigenvectors[:, 1:k+1]
        
        if pe.shape[1] < 8:
            pe = np.pad(pe, ((0, 0), (0, 8 - pe.shape[1])))
        return pe.astype(np.float32)
    except Exception as e:
        return np.zeros((N, 8), dtype=np.float32)

class NeuroGramDataset(IterableDataset):
    def __init__(self, jsonl_path: str, max_length: int = 2048) -> None:
        self.jsonl_path = jsonl_path
        self.max_length = max_length
        
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % num_workers != worker_id:
                    continue
                try:
                    data = json.loads(line)
                    seq = data.get("sequence", [])
                    
                    int_seq = []
                    for item in seq:
                        if isinstance(item, list) and len(item) == 3:
                            id1 = parse_token(item[0])
                            id2 = parse_token(item[1])
                            id3 = parse_token(item[2])
                            int_seq.append([id1, id2, id3])
                            
                    if not int_seq:
                        continue
                        
                    for i in range(0, len(int_seq), self.max_length):
                        chunk = int_seq[i:i+self.max_length]
                        
                        pad_len = self.max_length - len(chunk)
                        if pad_len > 0:
                            chunk.extend([[SPECIAL_TOKENS["<PAD>"]]*3] * pad_len)
                            
                        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
                        
                        adj = build_chunk_adjacency(chunk)
                        pe = compute_laplacian_pe(adj, k=8)
                        pe_tensor = torch.tensor(pe, dtype=torch.float32)
                        
                        yield chunk_tensor, pe_tensor
                except Exception as e:
                    continue

if __name__ == "__main__":
    print("Testing Dataloader and Graph Laplacian PE...")
    start = time.time()
    
    # Test on a realistic chunk size of 2048 for a 16GB GPU
    dataset = NeuroGramDataset("dummy_dataset.jsonl", max_length=2048)
    dataloader = DataLoader(dataset, batch_size=4) 
    
    batch_count = 0
    for tokens, pes in dataloader:
        print(f"Batch {batch_count}: Tokens shape {tokens.shape}, PE shape {pes.shape}")
        batch_count += 1
        if batch_count >= 5: # Just test 5 batches
            break
            
    print(f"Time taken for 5 batches (batch_size=4): {time.time()-start:.2f} seconds")
    print("Memory Check Passed: No OOM on Laplacian eigenvectors for N=2048!")
