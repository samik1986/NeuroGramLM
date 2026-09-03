"""
Author: samik1986
Date: 2026-09-03
"""
import torch
import torch.nn as nn
import sys
import os

class DomainAdaptationKernel(nn.Module):
    """
    A high-dimensional kernel transformation applied post-fusion.
    Used exclusively during finetuning to adapt to a new biological domain 
    without destroying the pre-trained CCFv3 latent geometry.
    """
    def __init__(self, d_model):
        super().__init__()
        self.kernel = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x):
        # Residual connection
        return x + self.kernel(x)

class FinetuningWrapper(nn.Module):
    """
    Wraps the core NeuroGramLM architecture.
    Freezes all encoder layers (Geometric, Topological, Bio, and Fusion),
    injects the DomainAdaptationKernel over the encoder_memory,
    and allows gradients only for the kernel and the Decoders.
    """
    def __init__(self, base_model, d_model):
        super().__init__()
        self.base_model = base_model
        self.domain_kernel = DomainAdaptationKernel(d_model)
        
        # 1. Freeze all parameters in the base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze the decoders for adaptation
        for param in self.base_model.geom_decoder.parameters():
            param.requires_grad = True
        for param in self.base_model.topo_decoder.parameters():
            param.requires_grad = True
            
        # The domain_kernel is naturally unfrozen because it was just initialized.

    def forward(self, batch, targets=None, padding_mask=None, tgt_mask=None):
        """
        Mimics the forward pass of the base model but intercepts and transforms 
        the fused encoder_memory before passing it to the decoders.
        """
        # Part 1: Run the frozen Encoders
        # Extract inputs
        vq_ids = batch['vq_ids']
        topological_ids = batch['topological_ids']
        bio_volumes = batch.get('bio_volumes', None)

        geom_enc = self.base_model.geom_emb(vq_ids)
        geom_enc = self.base_model.geom_tower(geom_enc, padding_mask)

        topo_enc = self.base_model.topo_emb(topological_ids['strahler_order'], topological_ids['wl_hash'])
        topo_enc = self.base_model.topo_tower(topo_enc, padding_mask)

        if bio_volumes is not None:
            bio_enc = self.base_model.bio_emb(bio_volumes)
            bio_enc = self.base_model.bio_tower(bio_enc, padding_mask)
        else:
            bio_enc = torch.zeros_like(geom_enc)

        encoder_memory = self.base_model.fusion_block(geom_enc, topo_enc, padding_mask)
        final_memory = encoder_memory + bio_enc 

        # --- INTERCEPT: Apply High-Dimensional Kernel Transformation for New Domain ---
        final_memory = self.domain_kernel(final_memory)
        # ----------------------------------------------------------------------------

        # Part 2: Run the unfrozen Decoders
        if targets is not None:
            tgt_geom_x = self.base_model.geom_emb(targets['vq_ids_shifted'])
            geom_out = self.base_model.geom_decoder(tgt_geom_x, final_memory, tgt_mask, padding_mask)

            tgt_topo_x = self.base_model.topo_emb(targets['topological_ids_shifted']['strahler_order'], 
                                                  targets['topological_ids_shifted']['wl_hash'])
            topo_out = self.base_model.topo_decoder(tgt_topo_x, final_memory, tgt_mask, padding_mask)
        else:
            # During inference (if generating)
            geom_out = None
            topo_out = None

        outputs = {
            'encoder_memory': final_memory, # Return transformed memory
            'geom_out': geom_out,
            'topo_out': topo_out
        }

        # Loss calculation uses the base model's logic
        if targets is not None:
            loss = 0.0
            loss += self.base_model.geom_loss(geom_out, targets['vq_ids'])
            loss += self.base_model.topo_loss(topo_out, targets['topological_ids'])
            loss += self.base_model.fusion_loss(final_memory, geom_enc, topo_enc)
            
            geom_logits = self.base_model.geom_decoder.out_proj(geom_out)
            loss += self.base_model.decoder_loss(geom_logits, targets['vq_ids']['tortuosity']) # Simplified
            outputs['loss'] = loss
            
        return outputs
