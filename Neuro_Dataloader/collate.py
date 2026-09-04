import torch
import os
import json
from torch.nn.utils.rnn import pad_sequence

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    COLLATE_CONFIG = json.load(f)['collate_parameters']

def neuro_collate_fn(batch):
    """
    Collate function for NeuroDataset.
    Pads variable length sequences to the max length in the batch.
    Generates padding masks and causal target masks.
    """
    inputs_vq = {'tortuosity': [], 'curvature_energy': [], 'inertia_tensor': []}
    inputs_topo = {'strahler_order': [], 'wl_hash': []}
    
    targets_vq = {'tortuosity': [], 'curvature_energy': [], 'inertia_tensor': []}
    targets_topo = {'strahler_order': [], 'wl_hash': []}
    
    targets_shifted_vq = {'tortuosity': [], 'curvature_energy': [], 'inertia_tensor': []}
    targets_shifted_topo = {'strahler_order': [], 'wl_hash': []}
    
    lengths = []
    
    for item in batch:
        inp = item['inputs']
        tgt = item['targets']
        
        # We assume length of all keys in a sample is the same
        seq_len = inp['vq_ids']['tortuosity'].size(0)
        lengths.append(seq_len)
        
        for k in inputs_vq.keys():
            inputs_vq[k].append(inp['vq_ids'][k])
            targets_vq[k].append(tgt['vq_ids'][k])
            targets_shifted_vq[k].append(tgt['vq_ids_shifted'][k])
            
        for k in inputs_topo.keys():
            inputs_topo[k].append(inp['topological_ids'][k])
            targets_topo[k].append(tgt['topological_ids'][k])
            targets_shifted_topo[k].append(tgt['topological_ids_shifted'][k])
            
    # Pad sequences
    # Using padding_idx based on COLLATE_CONFIG
    
    def pad_dict(d, pad_val):
        return {k: pad_sequence(v, batch_first=True, padding_value=pad_val) for k, v in d.items()}
        
    pad_in = COLLATE_CONFIG.get('padding_idx_inputs', 0)
    pad_tgt = COLLATE_CONFIG.get('padding_idx_targets', -1)
    
    batched_inputs_vq = pad_dict(inputs_vq, pad_in)
    batched_inputs_topo = pad_dict(inputs_topo, pad_in)
    
    batched_targets_vq = pad_dict(targets_vq, pad_tgt)
    batched_targets_topo = pad_dict(targets_topo, pad_tgt)
    
    batched_targets_shifted_vq = pad_dict(targets_shifted_vq, 0)
    batched_targets_shifted_topo = pad_dict(targets_shifted_topo, 0)
    
    # Generate padding mask for Encoder
    # Shape should be (batch, seq_len) where True means padding token
    batch_size = len(batch)
    max_len = max(lengths)
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        padding_mask[i, l:] = True
        
    # Generate causal mask for Decoder
    # Shape: (max_len, max_len)
    tgt_mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
    
    # Pack into the expected batch structure
    # NOTE: The padding_mask and tgt_mask are placed in the root of the returned dict,
    # and the training script will be updated to correctly extract them.
    
    collated_batch = {
        'inputs': {
            'vq_ids': batched_inputs_vq,
            'topological_ids': batched_inputs_topo
        },
        'targets': {
            'vq_ids': batched_targets_vq,
            'topological_ids': batched_targets_topo,
            'vq_ids_shifted': batched_targets_shifted_vq,
            'topological_ids_shifted': batched_targets_shifted_topo
        },
        'padding_mask': padding_mask,
        'tgt_mask': tgt_mask
    }
    
    return collated_batch
