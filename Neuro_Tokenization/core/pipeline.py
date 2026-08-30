"""
Author: Samik Banerjee
Date: 2026-08-30
Usage: Main tokenization pipeline orchestrator to extract features, assemble tokens, and evaluate quality.
"""

import logging
from .features import compute_tortuosity, compute_curvature_energy, compute_local_inertia_tensor
from .tokenizer import MultimodalTokenizer
from .quality import evaluate_token_quality
import numpy as np

# Configure logging for step tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class TokenizationPipeline:
    def __init__(self, config):
        self.config = config
        self.tokenizer = MultimodalTokenizer(config)
        
        # Initialize and load VQ codebooks if available
        from .vq import MultimodalVQ
        self.vq = MultimodalVQ(config)
        self.vq.load()
        
        logging.info("TokenizationPipeline initialized.")
        
    def extract_features(self, points):
        """Runs the mathematical feature extraction step."""
        logging.info(f"Extracting features for {len(points)} points...")
        
        # 1. Tortuosity
        w_size = self.config['algorithm_parameters'].get('tortuosity_window_size', 5)
        tortuosity = compute_tortuosity(points, window_size=w_size)
        
        # 2. Curvature Energy
        sigma = self.config['algorithm_parameters'].get('curvature_smoothing_sigma', 2.0)
        curvature = compute_curvature_energy(points, sigma=sigma)
        
        # 3. Inertia Tensor (using the first point as query for simplicity in this fragment demo)
        # In a real scenario, this would be computed per node
        radius = self.config['algorithm_parameters'].get('inertia_radius', 10.0)
        inertia_tensors = [compute_local_inertia_tensor(points, p, radius) for p in points]
        
        features = {
            'tortuosity': tortuosity,
            'curvature_energy': curvature,
            'inertia_tensor': inertia_tensors
            # branching_angle would require tree topology info, skipped in this linear demo
        }
        
        logging.info("Feature extraction complete.")
        return features
        
    def process_fragment(self, points, available_modalities=['tortuosity', 'curvature_energy', 'inertia_tensor']):
        """
        Main pipeline function. 
        Takes a sequence of 3D coordinates (points) representing a fragment.
        """
        try:
            # Step 1: Feature Extraction
            features = self.extract_features(points)
            
            # Step 2: Tokenization & Masking
            logging.info("Assembling multimodal tokens...")
            nodes = [{'id': i} for i in range(len(points))]
            tokens = self.tokenizer.tokenize_fragment(nodes, features, available_modalities)
            
            # Step 2.5: Multimodal Vector Quantization
            if self.vq.is_trained:
                for token in tokens:
                    token['vq_ids'] = self.vq.quantize(token['embeddings'])
            else:
                for token in tokens:
                    token['vq_ids'] = {}
            
            # Step 3: Quality Evaluation
            logging.info("Evaluating token quality...")
            score, metrics = evaluate_token_quality(tokens, self.config)
            
            if metrics['is_valid']:
                logging.info(f"Pipeline succeeded. Quality Score: {score:.2f}")
            else:
                logging.warning(f"Pipeline produced low-quality tokens. Penalties: {metrics['penalties']}")
                
            return {
                'success': True,
                'tokens': tokens,
                'quality_score': score,
                'metrics': metrics
            }
            
        except Exception as e:
            logging.error(f"Pipeline failed during processing: {str(e)}")
            return {
                'success': False,
                'error_log': str(e)
            }
            
    def append_modality(self, tokens, name, values):
        """Utility to append a new modality to existing tokens after the fact."""
        if len(tokens) != len(values):
            raise ValueError("Values array length must match tokens list length.")
            
        self.tokenizer.register_modality(name, dim=np.array(values[0]).size)
        
        for i, token in enumerate(tokens):
            token['embeddings'][name] = values[i]
            token['attention_masks'][name] = 1.0
            
        logging.info(f"Successfully appended new modality: '{name}'")
        return tokens
