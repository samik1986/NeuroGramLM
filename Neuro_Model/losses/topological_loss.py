"""
Author: samik1986
Date: 2026-09-03
"""
import torch
import torch.nn as nn

class TopologicalLoss(nn.Module):
    """
    Loss module for the Topological tower.
    Computes Cross-Entropy for predicting the topological indices.
    """
    def __init__(self, vocab_sizes):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
    def forward(self, logits_dict, targets_dict):
        """
        logits_dict: Dict of tensors for 'strahler_order', 'wl_hash'
        targets_dict: Dict of target integers for the same keys
        """
        loss = 0.0
        for mod, logits in logits_dict.items():
            if mod in targets_dict:
                targets = targets_dict[mod]
                loss += self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
        return loss
