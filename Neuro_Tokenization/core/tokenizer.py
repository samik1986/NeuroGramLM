"""
Author: Samik Banerjee
Date: 2026-08-30
Usage: MultimodalTokenizer to assemble features and handle missing modalities.
"""

import numpy as np

class MultimodalTokenizer:
    def __init__(self, config):
        self.config = config
        self.mask_value = config['tokenization_parameters'].get('mask_modality_value', 0.0)
        
        # Define the base modalities and their dimensions
        self.modality_registry = {
            'tortuosity': 1,
            'curvature_energy': 1,
            'inertia_tensor': 3, # 3 eigenvalues
            'branching_angle': 1,
            'strahler_order': 1,
            'wl_hash': 1,
            'background_intensity': 1
        }
        
    def register_modality(self, name, dim):
        """Allows dynamic appending of new modalities."""
        self.modality_registry[name] = dim
        
    def assemble_token(self, node_id, features, available_modalities):
        """
        Assembles a single multimodal token.
        If a modality in the registry is not in available_modalities, it is masked.
        """
        token = {'node_id': node_id, 'embeddings': {}, 'attention_masks': {}}
        
        for mod_name, mod_dim in self.modality_registry.items():
            if mod_name in available_modalities and mod_name in features:
                val = features[mod_name]
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                elif isinstance(val, (np.floating, np.integer)):
                    val = val.item()
                token['embeddings'][mod_name] = val
                token['attention_masks'][mod_name] = 1.0 # 1 means available
            else:
                # Modality masking
                if mod_dim == 1:
                    token['embeddings'][mod_name] = self.mask_value
                else:
                    token['embeddings'][mod_name] = [self.mask_value] * mod_dim
                token['attention_masks'][mod_name] = 0.0 # 0 means masked
                
        return token

    def tokenize_fragment(self, nodes, node_features, available_modalities):
        """
        Takes a sequence of nodes (e.g., from a Depth-First Traversal of a fragment)
        and their computed features, and returns a list of tokens.
        """
        tokens = []
        for i, node in enumerate(nodes):
            # Extract features for this specific node
            n_features = {k: v[i] for k, v in node_features.items()}
            
            token = self.assemble_token(
                node_id=node.get('id', i), 
                features=n_features, 
                available_modalities=available_modalities
            )
            tokens.append(token)
            
        return tokens
