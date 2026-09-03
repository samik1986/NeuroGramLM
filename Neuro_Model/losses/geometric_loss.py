import torch
import torch.nn as nn

class GeometricLoss(nn.Module):
    """
    Loss module for the Geometric/Morphological tower.
    Computes Cross-Entropy for predicting the VQ discrete IDs.
    """
    def __init__(self, vocab_sizes):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_sizes = vocab_sizes
        
        # We need projection heads if we are calculating auxiliary loss directly from the tower
        # For simplicity, these could be set externally, or defined here if dimensions are known.

    def forward(self, logits_dict, targets_dict):
        """
        logits_dict: Dict of tensors for 'tortuosity', 'curvature_energy', 'inertia_tensor'
        targets_dict: Dict of target VQ IDs for the same keys
        """
        loss = 0.0
        for mod, logits in logits_dict.items():
            if mod in targets_dict:
                targets = targets_dict[mod]
                # Cross entropy expects (N, C, d1, d2, ...). 
                # Our logits are (batch, seq_len, vocab_size).
                # Reshape to (batch * seq_len, vocab_size) and (batch * seq_len,)
                loss += self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
        return loss
