"""
Author: Samik Banerjee
Date: 2026-08-30
Usage: Functions to evaluate biological validity and quality of tokens.
"""

import numpy as np

def evaluate_token_quality(tokens, config):
    """
    Evaluates a sequence of tokens to determine its biological validity and 
    information richness. Returns a quality score between 0.0 and 1.0.
    """
    thresholds = config['quality_thresholds']
    
    if len(tokens) < thresholds['min_nodes_per_fragment']:
        return 0.0, {
            'is_valid': False,
            'penalties': ["Fragment too short"],
            'modality_completeness': 0.0
        }
        
    score = 1.0
    penalties = []
    
    # Check Modality Completeness
    total_modalities = 0
    present_modalities = 0
    for token in tokens:
        for mod, mask in token['attention_masks'].items():
            total_modalities += 1
            if mask == 1.0:
                present_modalities += 1
                
    modality_ratio = present_modalities / max(1, total_modalities)
    # Penalize if too many modalities are missing (masking)
    if modality_ratio < 0.5:
        score -= 0.3
        penalties.append(f"Low modality completeness: {modality_ratio:.2f}")
        
    # Check Biological Plausibility (e.g., Extreme Tortuosity)
    tortuosities = [t['embeddings']['tortuosity'] for t in tokens if t['attention_masks'].get('tortuosity', 0) == 1.0]
    if tortuosities:
        max_t = np.max(tortuosities)
        if max_t > thresholds['max_tortuosity_allowed']:
            score -= 0.4
            penalties.append(f"Unrealistic tortuosity found: {max_t:.2f}")
            
    # Normalize score
    score = max(0.0, score)
    
    # Decision
    is_valid = score >= thresholds['min_biological_score']
    
    return score, {
        'is_valid': is_valid,
        'penalties': penalties,
        'modality_completeness': modality_ratio
    }
