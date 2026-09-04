import os
import torch
import glob
from torch.utils.data import DataLoader
from Neuro_Dataloader.dataset import NeuroDataset
from Neuro_Dataloader.collate import neuro_collate_fn
from Neuro_Model.model import NeuroGramLM
import json

def test():
    print("Testing Dataloader and Model forward pass...")
    config_path = 'Neuro_Training/config.json'
    model_config_path = 'Neuro_Model/config.json'
    
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
        
    token_dir = os.path.abspath(os.path.join('Neuro_Training', train_config['io_paths']['token_input_dir']))
    all_files = glob.glob(os.path.join(token_dir, "*.json"))[:10]  # Just take 10 files
    
    dataset = NeuroDataset(token_dir, file_list=all_files)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=neuro_collate_fn, num_workers=0)
    
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input padding mask shape: {batch['padding_mask'].shape}")
    print(f"Target mask shape: {batch['tgt_mask'].shape}")
    
    device = torch.device('cpu')
    model = NeuroGramLM(model_config).to(device)
    model.eval()
    
    padding_mask = batch.get('padding_mask', None)
    tgt_mask = batch.get('tgt_mask', None)

    inputs = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
             for k, v in batch['inputs'].items()}
    targets = {k: {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v.to(device) 
               for k, v in batch['targets'].items()}
               
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(inputs, targets=targets, padding_mask=padding_mask, tgt_mask=tgt_mask)
        
    print(f"Loss: {outputs['loss'].item()}")
    print("Success! No runtime errors.")

if __name__ == "__main__":
    test()
