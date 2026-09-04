import torch
import os
import json
import glob
from torch.utils.data import DataLoader
from Neuro_Dataloader.dataset import NeuroDataset
from Neuro_Dataloader.collate import neuro_collate_fn

token_dir = "/mnt/diskg9-3/NeuroGramLM/Neuro_Tokenization/data/tokenized_output"
files = glob.glob(os.path.join(token_dir, "*.json"))
print(f"Found {len(files)} files.")
dataset = NeuroDataset(token_dir, file_list=files[:500])
loader = DataLoader(dataset, batch_size=128, collate_fn=neuro_collate_fn)

for i, batch in enumerate(loader):
    for k, v in batch['inputs']['vq_ids'].items():
        if v.min() < 0 or v.max() >= 256: 
            if k == 'inertia_tensor' and v.max() < 512: continue
            print(f"ERROR: {k} input {v.min()} - {v.max()}")
    for k, v in batch['inputs']['topological_ids'].items():
        if v.min() < 0 or (k == 'strahler_order' and v.max() >= 50) or (k == 'wl_hash' and v.max() >= 1000):
            print(f"ERROR: {k} topo {v.min()} - {v.max()}")
    for k, v in batch['targets']['vq_ids'].items():
        if v.min() < -1 or v.max() >= 256: 
            if k == 'inertia_tensor' and v.max() < 512: continue
            print(f"ERROR: target {k} {v.min()} - {v.max()}")
    print(f"Batch {i} passed check.")
