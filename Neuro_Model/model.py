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

    def forward(self, batch, targets=None, padding_mask=None, tgt_mask=None):
        """
        batch: Dict containing 'vq_ids', 'topological_ids', and optionally 'bio_volumes'
        targets: Target dict for teacher-forced autoregressive decoding
        """
        # ================= ENCODER =================
        geom_x = self.geom_emb(batch['vq_ids'])
        topo_x = self.topo_emb(batch['topological_ids']['strahler_order'], 
                               batch['topological_ids']['wl_hash'])
        
        geom_enc = self.geom_tower(geom_x, mask=padding_mask)
        topo_enc = self.topo_tower(topo_x, mask=padding_mask)
        
        # Zero-shot Bio Encoding (If provided at test time, otherwise zeroed/dropout)
        if 'bio_volumes' in batch and batch['bio_volumes'] is not None:
            bio_x = self.bio_emb(batch['bio_volumes'])
            bio_enc = self.bio_tower(bio_x, mask=padding_mask)
        else:
            # During training, Bio is effectively null
            bio_enc = torch.zeros_like(geom_enc)
            
        # Fusion Memory (Geometry + Topology)
        encoder_memory = self.fusion_block(geom_enc, topo_enc, mask=padding_mask)
        
        # In a real forward pass, we might concatenate bio_enc to encoder_memory at test time, 
        # or use dual cross-attention in the decoder. For simplicity here, we add it to the memory
        # since it's zeroed during training and won't affect gradients.
        final_memory = encoder_memory + bio_enc 
        
        # ================= DECODER =================
        # If we have shifted targets for teacher forcing
        if targets is not None:
            tgt_geom_x = self.geom_emb(targets['vq_ids_shifted'])
            tgt_topo_x = self.topo_emb(targets['topological_ids_shifted']['strahler_order'],
                                       targets['topological_ids_shifted']['wl_hash'])
                                       
            dec_geom_out = self.geom_decoder(tgt_geom_x, final_memory, tgt_mask=tgt_mask)
            dec_topo_out = self.topo_decoder(tgt_topo_x, final_memory, tgt_mask=tgt_mask)
            
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
            
            # Calculate Decoder Losses
            l_dec_geom = self.decoder_loss_fn(geom_logits, targets['vq_ids'])
            l_dec_topo = self.decoder_loss_fn(topo_logits, targets['topological_ids'])
            
            total_loss = l_dec_geom + l_dec_topo
            
            return {
                'loss': total_loss,
                'geom_logits': geom_logits,
                'topo_logits': topo_logits
            }
        else:
            # Inference mode (would implement iterative generation loop here)
            return {'encoder_memory': final_memory}
