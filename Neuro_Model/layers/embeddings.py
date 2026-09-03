import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    Applies sequential positional encoding to variable length fragments.
    """
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Calculate inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for sin and cos
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, x, seq_len):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.shape[2]
            
        self._update_cos_sin_cache(q, seq_len)
        
        cos = self._cos_cached[:, :, :seq_len, ...].to(q.device)
        sin = self._sin_cached[:, :, :seq_len, ...].to(q.device)
        
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed

class TopologicalEmbedding(nn.Module):
    def __init__(self, strahler_vocab_size, wl_vocab_size, dim):
        super().__init__()
        self.strahler_emb = nn.Embedding(strahler_vocab_size, dim)
        self.wl_emb = nn.Embedding(wl_vocab_size, dim)
        
    def forward(self, strahler_indices, wl_indices):
        s_emb = self.strahler_emb(strahler_indices)
        w_emb = self.wl_emb(wl_indices)
        return s_emb + w_emb

class GeometricEmbedding(nn.Module):
    def __init__(self, vocab_sizes, dim):
        super().__init__()
        self.tortuosity_emb = nn.Embedding(vocab_sizes['tortuosity'], dim)
        self.curvature_emb = nn.Embedding(vocab_sizes['curvature_energy'], dim)
        self.inertia_emb = nn.Embedding(vocab_sizes['inertia_tensor'], dim)
        
    def forward(self, vq_ids):
        t_emb = self.tortuosity_emb(vq_ids['tortuosity'])
        c_emb = self.curvature_emb(vq_ids['curvature_energy'])
        i_emb = self.inertia_emb(vq_ids['inertia_tensor'])
        return t_emb + c_emb + i_emb

class BioEmbedding(nn.Module):
    """
    3D CNN Embedding block for Biological Intensity Data (Zero-shot inference).
    Processes volumetric intensity patches around a node/ridge.
    """
    def __init__(self, cnn_config, d_model):
        super().__init__()
        in_c = cnn_config.get('in_channels', 1)
        out_c = cnn_config.get('out_channels', 64)
        k_size = cnn_config.get('kernel_size', 3)
        stride = cnn_config.get('stride', 1)
        pad = cnn_config.get('padding', 1)
        
        self.cnn3d = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=k_size, stride=stride, padding=pad),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(out_c, out_c * 2, kernel_size=k_size, stride=stride, padding=pad),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1) # Flatten spatial dims
        )
        self.projection = nn.Linear(out_c * 2, d_model)

    def forward(self, volume_patches):
        """
        volume_patches: (batch, seq_len, in_c, D, H, W)
        """
        batch, seq_len, c, d, h, w = volume_patches.shape
        # Flatten batch and seq_len for CNN
        flat_vols = volume_patches.view(batch * seq_len, c, d, h, w)
        
        features = self.cnn3d(flat_vols) # (batch*seq_len, out_c*2, 1, 1, 1)
        features = features.view(batch, seq_len, -1)
        
        return self.projection(features)
