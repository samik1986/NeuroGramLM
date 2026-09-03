"""
Author: samik1986
Date: 2026-09-03
"""
import torch
import torch.nn as nn

class DecoderLoss(nn.Module):
    """
    Loss module for the Decoder Multi-Tower.
    Computes autoregressive Cross-Entropy for generating the bridging tokens.
    """
    def __init__(self, vocab_sizes):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_sizes = vocab_sizes

    def forward(self, logits_dict, targets_dict):
        """
        logits_dict: Dict of tensors from decoder projection heads
        targets_dict: Dict of shifted target VQ/Topological IDs
        """
        loss = 0.0
        for mod, logits in logits_dict.items():
            if mod in targets_dict:
                targets = targets_dict[mod]
                loss += self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
        return loss
