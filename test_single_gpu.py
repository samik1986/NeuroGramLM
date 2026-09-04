import torch
import json
import os
from torch.cuda.amp import autocast, GradScaler
from Neuro_Model.model import NeuroGramLM
from Neuro_Dataloader.dataset import NeuroDataset
from Neuro_Dataloader.collate import neuro_collate_fn
from torch.utils.data import DataLoader
import glob
import torch.nn as nn

with open('Neuro_Model/config.json', 'r') as f:
    config = json.load(f)

model = NeuroGramLM(config).to('cuda:0')

token_dir = "/mnt/diskg9-3/NeuroGramLM/Neuro_Tokenization/data/tokenized_output"
files = glob.glob(os.path.join(token_dir, "*.json"))
dataset = NeuroDataset(token_dir, file_list=files[:1000])
loader = DataLoader(dataset, batch_size=128, collate_fn=neuro_collate_fn)
scaler = GradScaler(enabled=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("Starting single GPU test...")
for i, batch in enumerate(loader):
    inputs = {k: {sk: sv.to('cuda:0') for sk, sv in v.items()} if isinstance(v, dict) else v.to('cuda:0') for k, v in batch['inputs'].items()}
    targets = {k: {sk: sv.to('cuda:0') for sk, sv in v.items()} if isinstance(v, dict) else v.to('cuda:0') for k, v in batch['targets'].items()}
    padding_mask = batch['padding_mask'].to('cuda:0')
    
    optimizer.zero_grad()
    with autocast(device_type='cuda', enabled=True):
        outputs = model(inputs, targets=targets, padding_mask=padding_mask)
        loss = outputs['loss']
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Batch {i} success! Loss: {loss.item()}")
    if i == 5: break
print("Test completed.")
