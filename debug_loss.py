import torch
import json
import os
from Neuro_Model.model import NeuroGramLM
from Neuro_Dataloader.dataset import NeuroDataset
from Neuro_Dataloader.collate import neuro_collate_fn
from torch.utils.data import DataLoader
import glob
import torch.nn as nn

with open('Neuro_Model/config.json', 'r') as f:
    config = json.load(f)

model = NeuroGramLM(config)
model = model.to('cuda')

def make_loss_hook():
    def hook(module, input, output):
        logits = input[0]
        targets = input[1]
        valid_mask = targets != -100
        valid_targets = targets[valid_mask]
        
        if valid_targets.numel() > 0:
            t_max = valid_targets.max().item()
            t_min = valid_targets.min().item()
            if t_min < 0 or t_max >= logits.size(-1):
                print(f"!!! CRASH IN LOSS !!! logits size: {logits.size(-1)}, min: {t_min}, max: {t_max}")
                print(f"Invalid targets: {valid_targets[(valid_targets < 0) | (valid_targets >= logits.size(-1))]}")
    return hook

model.decoder_loss_fn.criterion.register_forward_hook(make_loss_hook())

token_dir = "/mnt/diskg9-3/NeuroGramLM/Neuro_Tokenization/data/tokenized_output"
files = glob.glob(os.path.join(token_dir, "*.json"))
dataset = NeuroDataset(token_dir, file_list=files[:500])
loader = DataLoader(dataset, batch_size=128, collate_fn=neuro_collate_fn)

print("Starting forward passes...")
for i, batch in enumerate(loader):
    inputs = {k: {sk: sv.to('cuda') for sk, sv in v.items()} if isinstance(v, dict) else v.to('cuda') for k, v in batch['inputs'].items()}
    targets = {k: {sk: sv.to('cuda') for sk, sv in v.items()} if isinstance(v, dict) else v.to('cuda') for k, v in batch['targets'].items()}
    padding_mask = batch['padding_mask'].to('cuda')
    
    try:
        model(inputs, targets=targets, padding_mask=padding_mask)
    except Exception as e:
        print(f"Caught exception on batch {i}: {e}")
        break
print("Done.")
