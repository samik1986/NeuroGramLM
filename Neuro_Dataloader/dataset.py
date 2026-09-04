import os
import json
import torch
import glob
from torch.utils.data import Dataset
import numpy as np

class NeuroDataset(Dataset):
    def __init__(self, token_dir, file_list=None):
        super().__init__()
        self.token_dir = token_dir
        
        # Load local config
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)['dataset_parameters']
            
        if file_list is not None:
            self.files = file_list
        else:
            self.files = glob.glob(os.path.join(token_dir, "*.json"))
            
        # We might not load everything into memory to avoid OOM
        # But for simplicity if the dataset is small enough we can store just file paths
        # and load them in __getitem__, but each file contains multiple fragments!
        # Actually a single file contains `tokens` array which represents a single continuous stream or fragments.
        # Let's load the index of all files.
        self.samples = []
        for f in self.files:
            self.samples.append(f)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        file_path = self.samples[idx]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    data = {}
                else:
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        # Attempt to extract first valid JSON object if extra data exists
                        decoder = json.JSONDecoder()
                        data, _ = decoder.raw_decode(content)
        except Exception:
            # Fallback to empty data structure on corrupted / unparseable file
            data = {}
            
        tokens = data.get('tokens', [])
        
        if len(tokens) < 2:
            if len(tokens) == 1:
                tokens = [tokens[0], tokens[0]]
            else:
                tokens = [{'vq_ids': {}, 'embeddings': {}}, {'vq_ids': {}, 'embeddings': {}}]
        
        # We need to construct tensors for vq_ids and topological_ids
        vq_tort = []
        vq_curv = []
        vq_iner = []
        
        topo_stra = []
        topo_wl = []
        
        # If the sequence is longer than max_seq_len, take a random slice
        max_seq_len = self.config.get('max_seq_len', 2048)
        if len(tokens) > max_seq_len:
            start_idx = np.random.randint(0, len(tokens) - max_seq_len)
            tokens = tokens[start_idx:start_idx + max_seq_len]
            
        null_val = self.config.get('null_imputation_value', 0)
        wl_vocab = self.config.get('wl_vocab_size', 1000)
            
        for t in tokens:
            vq = t.get('vq_ids', {})
            emb = t.get('embeddings', {})
            
            vq_tort.append(max(0, min(vq.get('tortuosity', null_val), 255)))
            vq_curv.append(max(0, min(vq.get('curvature_energy', null_val), 255)))
            vq_iner.append(max(0, min(vq.get('inertia_tensor', null_val), 511)))
            
            topo_stra.append(max(0, min(emb.get('strahler_order', null_val), 49)))
            wl = emb.get('wl_hash', null_val)
            if wl is None:
                wl = null_val
            topo_wl.append(max(0, wl % wl_vocab))
            
        inputs_vq = {
            'tortuosity': torch.tensor(vq_tort[:-1], dtype=torch.long),
            'curvature_energy': torch.tensor(vq_curv[:-1], dtype=torch.long),
            'inertia_tensor': torch.tensor(vq_iner[:-1], dtype=torch.long)
        }
        
        inputs_topo = {
            'strahler_order': torch.tensor(topo_stra[:-1], dtype=torch.long),
            'wl_hash': torch.tensor(topo_wl[:-1], dtype=torch.long)
        }
        
        targets_vq = {
            'tortuosity': torch.tensor(vq_tort[1:], dtype=torch.long),
            'curvature_energy': torch.tensor(vq_curv[1:], dtype=torch.long),
            'inertia_tensor': torch.tensor(vq_iner[1:], dtype=torch.long)
        }
        
        targets_topo = {
            'strahler_order': torch.tensor(topo_stra[1:], dtype=torch.long),
            'wl_hash': torch.tensor(topo_wl[1:], dtype=torch.long)
        }
        
        # We also need vq_ids_shifted and topological_ids_shifted as they are used in Decoder
        # Actually targets['vq_ids_shifted'] should be the inputs for the decoder (teacher forcing)
        # So targets['vq_ids_shifted'] is exactly inputs_vq, but padded with a start token?
        # In a standard autoregressive model, the decoder input is the shifted target sequence.
        # But wait, in model.py, `tgt_geom_x = self.geom_emb(targets['vq_ids_shifted'])` 
        # is used to predict `targets['vq_ids']`.
        # So `targets['vq_ids_shifted']` IS the input to the decoder, which is usually the same as `inputs_vq`
        # if the encoder and decoder share the same sequence (like in translation, encoder gets src, decoder gets tgt[:-1]).
        # But here it's an encoder-decoder LM. Wait, if it's predicting the sequence, the encoder gets the context (e.g. up to t),
        # but in standard autoregressive LM, you only need a decoder.
        # This is an encoder-decoder. Maybe the encoder takes the full corrupted sequence, and decoder reconstructs it?
        # The README says "trained on continuous neuron fragments using autoregressive teacher-forcing."
        # If it's a standard autoregressive model, we use the input sequence as `inputs` and `targets_shifted` (which is the same),
        # and `targets` is offset by 1.
        
        return {
            'inputs': {
                'vq_ids': inputs_vq,
                'topological_ids': inputs_topo
            },
            'targets': {
                'vq_ids': targets_vq,
                'topological_ids': targets_topo,
                'vq_ids_shifted': inputs_vq,
                'topological_ids_shifted': inputs_topo
            }
        }
