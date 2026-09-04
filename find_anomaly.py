import torch
import os
import json
import glob
from Neuro_Dataloader.dataset import NeuroDataset
from Neuro_Dataloader.collate import neuro_collate_fn
from torch.utils.data import DataLoader

token_dir = "/mnt/diskg9-3/NeuroGramLM/Neuro_Tokenization/data/tokenized_output"
files = glob.glob(os.path.join(token_dir, "*.json"))
dataset = NeuroDataset(token_dir, file_list=files[:1000])
loader = DataLoader(dataset, batch_size=128, collate_fn=neuro_collate_fn)

for i, batch in enumerate(loader):
    for k, v in batch['targets']['vq_ids_shifted'].items():
        if v.min() < 0 or v.max() >= (512 if k == 'inertia_tensor' else 256):
            print(f"ANOMALY in targets vq_ids_shifted {k}: min={v.min()}, max={v.max()}")
    for k, v in batch['targets']['topological_ids_shifted'].items():
        if v.min() < 0 or v.max() >= (1000 if k == 'wl_hash' else 50):
            print(f"ANOMALY in targets topological_ids_shifted {k}: min={v.min()}, max={v.max()}")
    for k, v in batch['inputs']['vq_ids'].items():
        if v.min() < 0 or v.max() >= (512 if k == 'inertia_tensor' else 256):
            print(f"ANOMALY in inputs vq_ids {k}: min={v.min()}, max={v.max()}")
print("Done finding anomalies on CPU.")
