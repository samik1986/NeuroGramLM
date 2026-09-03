import torch
import torch.nn as nn

class FusionLoss(nn.Module):
    """
    Specialized Loss module applied AFTER the multimodal fusion block.
    This can be used to optimize the shared latent space representation
    directly for high-level downstream tasks, such as next-token autoregressive
    prediction across all modalities jointly, or contrastive learning.
    """
    def __init__(self):
        super().__init__()
        # For demonstration, a placeholder MSE loss if predicting a continuous latent target
        # or it can be a combined CrossEntropy over a fused vocabulary.
        self.criterion = nn.MSELoss()

    def forward(self, fused_latents, targets):
        """
        fused_latents: (batch, seq_len, d_model)
        targets: (batch, seq_len, d_model)
        """
        return self.criterion(fused_latents, targets)
