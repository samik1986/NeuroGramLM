import torch
import json
import os
from Neuro_Model.model import NeuroGramLM
from Neuro_Dataloader.dataset import NeuroDataset
from Neuro_Dataloader.collate import neuro_collate_fn
from torch.utils.data import DataLoader
import glob

with open('Neuro_Model/config.json', 'r') as f:
    config = json.load(f)

model = NeuroGramLM(config)
model = model.to('cuda')

def make_hook(name):
    def hook(module, input, output):
        inp = input[0]
        if inp.min() < 0 or inp.max() >= module.num_embeddings:
            print(f"!!! CRASH IN {name} !!! min: {inp.min()}, max: {inp.max()}, num_emb: {module.num_embeddings}")
            print(f"Tensor shape: {inp.shape}")
            # print the first out of bounds value
            invalid = (inp < 0) | (inp >= module.num_embeddings)
            print(f"Invalid values: {inp[invalid]}")
    return hook

model.geom_emb.tortuosity_emb.register_forward_hook(make_hook('tortuosity_emb'))
model.geom_emb.curvature_emb.register_forward_hook(make_hook('curvature_emb'))
model.geom_emb.inertia_emb.register_forward_hook(make_hook('inertia_emb'))
model.topo_emb.strahler_emb.register_forward_hook(make_hook('strahler_emb'))
model.topo_emb.wl_emb.register_forward_hook(make_hook('wl_emb'))

token_dir = "/mnt/diskg9-3/NeuroGramLM/Neuro_Tokenization/data/tokenized_output"
files = glob.glob(os.path.join(token_dir, "*.json"))
dataset = NeuroDataset(token_dir, file_list=files[:500])
loader = DataLoader(dataset, batch_size=128, collate_fn=neuro_collate_fn)

print("Starting forward passes...")
for i, batch in enumerate(loader):
    inputs = {k: {sk: sv.to('cuda') for sk, sv in v.items()} if isinstance(v, dict) else v.to('cuda') for k, v in batch['inputs'].items()}
    targets = {k: {sk: sv.to('cuda') for sk, sv in v.items()} if isinstance(v, dict) else v.to('cuda') for k, v in batch['targets'].items()}
    padding_mask = batch['padding_mask'].to('cuda')
    
    # Try forward pass
    try:
        model(inputs, targets=targets, padding_mask=padding_mask)
    except Exception as e:
        print(f"Caught exception on batch {i}: {e}")
        break
print("Done.")
