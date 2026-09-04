"""
Author: samik1986
Date: 2026-09-03
"""
import torch
import torch.nn as nn
from .layers.embeddings import GeometricEmbedding, TopologicalEmbedding, BioEmbedding
from .layers.towers import GeometricTower, TopologicalTower, BioTower
from .layers.fusion import MultimodalFusion
from .layers.decoders import GeometricDecoder, TopologicalDecoder
from .losses import GeometricLoss, TopologicalLoss, FusionLoss, DecoderLoss

class NeuroGramLM(nn.Module):
    """
    Full Encoder-Decoder Architecture with Zero-Shot Bio Tower support.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        arch = config['architecture_parameters']
        
        d_model = arch['d_model']
        n_heads = arch['n_heads']
        dropout = arch['dropout']
        vocab_sizes = arch['vocab_sizes']
        
        # 1. Modality Embeddings
        self.geom_emb = GeometricEmbedding(vocab_sizes, d_model)
        self.topo_emb = TopologicalEmbedding(
            vocab_sizes['strahler_order'], 
            vocab_sizes['wl_hash'], 
            d_model
        )
        self.bio_emb = BioEmbedding(arch.get('cnn_3d', {}), d_model)
        
        # 2. Multi-Tower Encoders
        self.geom_tower = GeometricTower(d_model, n_heads, arch['n_layers_per_encoder'], dropout)
        self.topo_tower = TopologicalTower(d_model, n_heads, arch['n_layers_per_encoder'], dropout)
        self.bio_tower = BioTower(d_model, n_heads, arch['n_layers_per_encoder'], dropout)
        
        # 3. Multimodal Fusion (Encoder Memory)
        self.fusion_block = MultimodalFusion(d_model, n_heads, arch['n_layers_fusion'], dropout)
        
        # 4. Multi-Tower Decoders (Autoregressive Generation)
        self.geom_decoder = GeometricDecoder(d_model, n_heads, arch['n_layers_per_decoder'], dropout)
        self.topo_decoder = TopologicalDecoder(d_model, n_heads, arch['n_layers_per_decoder'], dropout)
        
        # 5. Output Heads (Decoders)
        self.geom_head_tort = nn.Linear(d_model, vocab_sizes['tortuosity'])
        self.geom_head_curv = nn.Linear(d_model, vocab_sizes['curvature_energy'])
        self.geom_head_iner = nn.Linear(d_model, vocab_sizes['inertia_tensor'])
        
        self.topo_head_stra = nn.Linear(d_model, vocab_sizes['strahler_order'])
        self.topo_head_wl = nn.Linear(d_model, vocab_sizes['wl_hash'])

        # Losses
        self.geom_loss_fn = GeometricLoss(vocab_sizes)
        self.topo_loss_fn = TopologicalLoss(vocab_sizes)
        self.fusion_loss_fn = FusionLoss()
        self.decoder_loss_fn = DecoderLoss(vocab_sizes)

    def forward(self, 
                vq_tortuosity, vq_curvature, vq_inertia, 
                topo_strahler, topo_wl,
                bio_volumes=None, 
                target_vq_tortuosity=None, target_vq_curvature=None, target_vq_inertia=None,
                target_topo_strahler=None, target_topo_wl=None,
                target_vq_tortuosity_shifted=None, target_vq_curvature_shifted=None, target_vq_inertia_shifted=None,
                target_topo_strahler_shifted=None, target_topo_wl_shifted=None,
                padding_mask=None, tgt_mask=None):
        """
        COMPLETELY flattened inputs to avoid DataParallel nested dictionary scatter corruption.
        """
        # ================= ENCODER =================
        vq_ids = {
            'tortuosity': vq_tortuosity,
            'curvature_energy': vq_curvature,
            'inertia_tensor': vq_inertia
        }
        geom_x = self.geom_emb(vq_ids)
        topo_x = self.topo_emb(topo_strahler, topo_wl)
        
        geom_enc = self.geom_tower(geom_x, mask=padding_mask)
        topo_enc = self.topo_tower(topo_x, mask=padding_mask)
        
        # Zero-shot Bio Encoding
        if bio_volumes is not None:
            bio_x = self.bio_emb(bio_volumes)
            bio_enc = self.bio_tower(bio_x, mask=padding_mask)
        else:
            bio_enc = torch.zeros_like(geom_enc)
            
        # Fusion Memory
        encoder_memory = self.fusion_block(geom_enc, topo_enc, mask=padding_mask)
        final_memory = encoder_memory + bio_enc 
        
        # ================= DECODER =================
        if target_vq_tortuosity_shifted is not None and target_topo_strahler_shifted is not None:
            targets_vq_shifted = {
                'tortuosity': target_vq_tortuosity_shifted,
                'curvature_energy': target_vq_curvature_shifted,
                'inertia_tensor': target_vq_inertia_shifted
            }
            targets_topo_shifted = {
                'strahler_order': target_topo_strahler_shifted,
                'wl_hash': target_topo_wl_shifted
            }
            
            # Replace -100 padding index with 0 for embedding lookups to prevent out of bounds memory access
            safe_targets_vq = {k: torch.where(v == -100, torch.zeros_like(v), v) for k, v in targets_vq_shifted.items()}
            safe_targets_topo = {k: torch.where(v == -100, torch.zeros_like(v), v) for k, v in targets_topo_shifted.items()}
            
            tgt_geom_x = self.geom_emb(safe_targets_vq)
            tgt_topo_x = self.topo_emb(safe_targets_topo['strahler_order'],
                                       safe_targets_topo['wl_hash'])
            
            # Generate causal mask internally
            seq_len = tgt_geom_x.size(1)
            internal_tgt_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=tgt_geom_x.device), diagonal=1)
            if tgt_mask is not None:
                internal_tgt_mask = internal_tgt_mask | tgt_mask
                                       
            dec_geom_out = self.geom_decoder(tgt_geom_x, final_memory, tgt_mask=internal_tgt_mask)
            dec_topo_out = self.topo_decoder(tgt_topo_x, final_memory, tgt_mask=internal_tgt_mask)
            
            # Predict Next Tokens
            geom_logits = {
                'tortuosity': self.geom_head_tort(dec_geom_out),
                'curvature_energy': self.geom_head_curv(dec_geom_out),
                'inertia_tensor': self.geom_head_iner(dec_geom_out)
            }
            topo_logits = {
                'strahler_order': self.topo_head_stra(dec_topo_out),
                'wl_hash': self.topo_head_wl(dec_topo_out)
            }
            
            # Reconstruct targets for loss
            targets_vq = {
                'tortuosity': target_vq_tortuosity,
                'curvature_energy': target_vq_curvature,
                'inertia_tensor': target_vq_inertia
            }
            targets_topo = {
                'strahler_order': target_topo_strahler,
                'wl_hash': target_topo_wl
            }
            
            # Calculate Decoder Losses
            l_dec_geom = self.decoder_loss_fn(geom_logits, targets_vq)
            l_dec_topo = self.decoder_loss_fn(topo_logits, targets_topo)
            
            total_loss = l_dec_geom + l_dec_topo
            
            if self.training:
                return total_loss
                
            return {
                'loss': total_loss,
                'geom_logits': geom_logits,
                'topo_logits': topo_logits
            }
        else:
            return {'encoder_memory': final_memory}
