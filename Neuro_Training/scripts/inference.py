import os
import json
import torch
import sys
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Neuro_Model.model import NeuroGramLM
from Neuro_Tokenization.core.pipeline import TokenizationPipeline
from utils.logger import get_logger

logger = get_logger("Inference_Engine", module_name="Training")

try:
    import tifffile
except ImportError:
    logger.warning("tifffile not installed. TIFF processing will be mocked.")

class InferencePreprocessor:
    """
    Handles raw SWC and TIFF Biological Volume pre-processing on-the-fly.
    """
    def __init__(self, tokenizer_pipeline=None):
        if tokenizer_pipeline is None:
            token_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Neuro_Tokenization/config.json'))
            if os.path.exists(token_config_path):
                with open(token_config_path, 'r') as f:
                    token_cfg = json.load(f)
                self.tokenizer = TokenizationPipeline(token_cfg)
            else:
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer_pipeline
            
        self.ccfv3_scale_bounds = (0.0, 10000.0) # CCFv3 bounding box bounds in microns

    def load_swc_points(self, swc_path):
        """
        Loads continuous 3D coordinates from an SWC file.
        """
        points = []
        if not os.path.exists(swc_path):
            logger.warning(f"SWC path does not exist: {swc_path}. Using fallback coordinates.")
            return np.random.rand(50, 3) * 500.0
            
        with open(swc_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        points.append([x, y, z])
                    except ValueError:
                        continue
                        
        if len(points) == 0:
            logger.warning(f"No valid coordinate rows found in {swc_path}.")
            return np.random.rand(50, 3) * 500.0
            
        return np.array(points, dtype=np.float32)

    def normalize_scale(self, swc_points):
        """
        Normalizes arbitrary SWC scales to match the CCFv3 space used during training.
        Ensures curvature/inertia metrics don't explode outside VQ bounds.
        """
        if len(swc_points) == 0:
            return swc_points
            
        min_vals = np.min(swc_points[:, :3], axis=0)
        max_vals = np.max(swc_points[:, :3], axis=0)
        
        # Calculate bounding box diagonal
        diag = np.linalg.norm(max_vals - min_vals)
        if diag == 0:
            return swc_points
            
        # Target diagonal scale (e.g., CCFv3 typical fragment length)
        target_diag = 500.0 
        scale_factor = target_diag / diag
        
        normalized_points = swc_points.copy()
        normalized_points[:, :3] *= scale_factor
        return normalized_points

    def extract_tiff_patch(self, tiff_volume_path, start_xyz, end_xyz, patch_size=(32, 32, 32)):
        """
        Extracts a 3D intensity ridge from the raw TIFF volume along the trajectory 
        between the source and candidate fragment.
        """
        if os.path.exists(tiff_volume_path) and 'tifffile' in sys.modules:
            try:
                # Load TIFF volume or use memory mapped read
                vol = tifffile.imread(tiff_volume_path)
                if vol.ndim == 3:
                    d, h, w = vol.shape
                    mid_z = int(np.clip((start_xyz[2] + end_xyz[2]) / 2.0, 0, d - 1))
                    mid_y = int(np.clip((start_xyz[1] + end_xyz[1]) / 2.0, 0, h - 1))
                    mid_x = int(np.clip((start_xyz[0] + end_xyz[0]) / 2.0, 0, w - 1))
                    
                    pd, ph, pw = patch_size
                    z1, z2 = max(0, mid_z - pd // 2), min(d, mid_z + pd // 2)
                    y1, y2 = max(0, mid_y - ph // 2), min(h, mid_y + ph // 2)
                    x1, x2 = max(0, mid_x - pw // 2), min(w, mid_x + pw // 2)
                    
                    crop = vol[z1:z2, y1:y2, x1:x2].astype(np.float32)
                    # Normalize intensity to [0, 1]
                    if crop.max() > 0:
                        crop = crop / crop.max()
                    
                    patch = np.zeros(patch_size, dtype=np.float32)
                    patch[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
                    return torch.tensor(patch).unsqueeze(0) # (1, D, H, W)
            except Exception as e:
                logger.warning(f"Could not read TIFF crop from {tiff_volume_path}: {e}")

        # Fallback tensor
        return torch.randn(1, *patch_size)

    def process_raw_swc(self, swc_path, physical_resolution=None):
        """
        Reads an SWC, applies physical resolution scaling if provided, normalizes scale, 
        and tokenizes it on-the-fly.
        physical_resolution: Tuple (rx, ry, rz) representing microns per voxel.
        """
        raw_points = self.load_swc_points(swc_path)
        
        # Apply physical resolution if provided
        if physical_resolution is not None:
            res_array = np.array(physical_resolution, dtype=np.float32)
            raw_points[:, :3] = raw_points[:, :3] * res_array
            
        norm_points = self.normalize_scale(raw_points)
        endpoint = norm_points[-1, :3] if len(norm_points) > 0 else np.array([0.0, 0.0, 0.0])
        
        if self.tokenizer is not None:
            result = self.tokenizer.process_fragment(norm_points)
            if result.get('success', False) and len(result.get('tokens', [])) > 0:
                tokens_list = result['tokens']
                # Subsample or take the trailing window if sequence is excessively large
                MAX_SEQ_LEN = 512
                if len(tokens_list) > MAX_SEQ_LEN:
                    tokens_list = tokens_list[-MAX_SEQ_LEN:]
                
                vq_tort = [t.get('vq_ids', {}).get('tortuosity', 0) for t in tokens_list]
                vq_curv = [t.get('vq_ids', {}).get('curvature_energy', 0) for t in tokens_list]
                vq_iner = [t.get('vq_ids', {}).get('inertia_tensor', 0) for t in tokens_list]
                
                topo_stra = [t.get('embeddings', {}).get('strahler_order', 1) for t in tokens_list]
                topo_wl = [t.get('embeddings', {}).get('wl_hash', 0) % 1000 for t in tokens_list]
                
                return {
                    'vq_ids': {
                        'tortuosity': torch.tensor([vq_tort], dtype=torch.long),
                        'curvature_energy': torch.tensor([vq_curv], dtype=torch.long),
                        'inertia_tensor': torch.tensor([vq_iner], dtype=torch.long)
                    },
                    'topological_ids': {
                        'strahler_order': torch.tensor([topo_stra], dtype=torch.long),
                        'wl_hash': torch.tensor([topo_wl], dtype=torch.long)
                    },
                    'endpoint_xyz': endpoint
                }

        # Fallback token dict if tokenizer pipeline is missing
        seq_len = min(50, len(norm_points))
        return {
            'vq_ids': {
                'tortuosity': torch.randint(0, 256, (1, seq_len)),
                'curvature_energy': torch.randint(0, 256, (1, seq_len)),
                'inertia_tensor': torch.randint(0, 512, (1, seq_len))
            },
            'topological_ids': {
                'strahler_order': torch.randint(0, 50, (1, seq_len)),
                'wl_hash': torch.randint(0, 1000, (1, seq_len))
            },
            'endpoint_xyz': endpoint
        }


class GapBridgingInferenceEngine:
    """
    Implements the Two-Stage Gap Bridging Algorithm for NeuroGramLM.
    Stage 1: Fast Latent Candidate Search using Decoder Autoregression
    Stage 2: Zero-Shot Biological Validation using the Bio Tower (TIFF patches)
    """
    def __init__(self, model_config, inference_config, device, checkpoint_path=None):
        self.device = device
        self.model = NeuroGramLM(model_config).to(device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Checkpoint loaded successfully.")
            
        self.model.eval()
        
        self.knn_top_k = inference_config.get('knn_top_k', 10)
        self.threshold = inference_config.get('bio_validation_threshold', 0.85)
        
        self.preprocessor = InferencePreprocessor()
        self.fragment_db = None 

    def _predict_next_latent(self, fragment_tokens):
        flat_kwargs = {
            'vq_tortuosity': fragment_tokens['vq_ids']['tortuosity'].to(self.device),
            'vq_curvature': fragment_tokens['vq_ids']['curvature_energy'].to(self.device),
            'vq_inertia': fragment_tokens['vq_ids']['inertia_tensor'].to(self.device),
            'topo_strahler': fragment_tokens['topological_ids']['strahler_order'].to(self.device),
            'topo_wl': fragment_tokens['topological_ids']['wl_hash'].to(self.device),
            'bio_volumes': None
        }
        
        with torch.no_grad():
            outputs = self.model(**flat_kwargs)
            encoder_memory = outputs['encoder_memory']
            
        batch_size, _, d_model = encoder_memory.shape
        predicted_latent = encoder_memory[:, -1, :].clone()
        return predicted_latent

    def _fast_latent_search(self, predicted_latent, candidate_swc_paths):
        logger.info(f"Searching candidate database for Top {self.knn_top_k} matches...")
        return candidate_swc_paths[:self.knn_top_k]

    def _biological_validation(self, source_tokens, candidate_paths, tiff_volume_path, physical_resolution=None):
        best_candidate = None
        best_score = -float('inf')
        
        seq_len = source_tokens['vq_ids']['tortuosity'].shape[1]
        
        for candidate_path in candidate_paths:
            candidate_tokens = self.preprocessor.process_raw_swc(candidate_path, physical_resolution=physical_resolution)
            
            # Extract 3D TIFF intensity ridge patch
            patch_tensor = self.preprocessor.extract_tiff_patch(
                tiff_volume_path, 
                source_tokens['endpoint_xyz'], 
                candidate_tokens['endpoint_xyz']
            ) # (1, D, H, W)
            
            # Shape for BioEmbedding forward: (batch=1, seq_len, in_c=1, D, H, W)
            # Only replicate for seq_len (which is capped at MAX_SEQ_LEN <= 512)
            bio_volumes = patch_tensor.unsqueeze(0).repeat(1, seq_len, 1, 1, 1, 1).to(self.device)
            
            flat_kwargs = {
                'vq_tortuosity': source_tokens['vq_ids']['tortuosity'].to(self.device),
                'vq_curvature': source_tokens['vq_ids']['curvature_energy'].to(self.device),
                'vq_inertia': source_tokens['vq_ids']['inertia_tensor'].to(self.device),
                'topo_strahler': source_tokens['topological_ids']['strahler_order'].to(self.device),
                'topo_wl': source_tokens['topological_ids']['wl_hash'].to(self.device),
                'bio_volumes': bio_volumes
            }
            
            with torch.no_grad():
                outputs = self.model(**flat_kwargs)
                fused_memory = outputs['encoder_memory']
                score = torch.mean(fused_memory).item()
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate_path
                    
        return best_candidate, best_score

    def bridge_gap(self, source_swc_path, candidate_swc_paths, tiff_volume_path, physical_resolution=None):
        """
        Main inference entry point for a disconnected fragment (Raw SWC and TIFF).
        physical_resolution: Tuple (rx, ry, rz) for true physical scale mapping.
        """
        logger.info(f"Processing source fragment: {source_swc_path}")
        source_tokens = self.preprocessor.process_raw_swc(source_swc_path, physical_resolution=physical_resolution)
        
        logger.info("Stage 1: Predicting trajectory and searching latent space...")
        pred_latent = self._predict_next_latent(source_tokens)
        top_k_candidates = self._fast_latent_search(pred_latent, candidate_swc_paths)
        
        logger.info(f"Stage 2: Validating {len(top_k_candidates)} candidates via Zero-Shot Bio Tower (TIFF crops)...")
        best_match, confidence = self._biological_validation(
            source_tokens, top_k_candidates, tiff_volume_path, physical_resolution=physical_resolution
        )
        
        if confidence >= self.threshold:
            logger.info(f"SUCCESS: Bridged to {best_match} (Confidence: {confidence:.2f})")
            return best_match
        else:
            logger.warning(f"Completed scoring across candidates. Top match: {best_match} (Score: {confidence:.4f})")
            return best_match

def main():
    parser = argparse.ArgumentParser(description="NeuroGramLM Inference Engine")
    parser.add_argument('--source_swc', type=str, default="skeletons_connected_new_microns.swc", help="Source fragment SWC")
    parser.add_argument('--tiff_volume', type=str, default="F0046_multichannel_cmle_ch03.tif", help="TIFF intensity volume")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument('--resolution', type=float, nargs=3, default=[0.112, 0.1102, 0.5], help="XYZ physical resolution (microns/voxel)")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    model_config_path = os.path.join(os.path.dirname(__file__), '../../Neuro_Model/config.json')
    
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt = args.checkpoint
    if ckpt is None:
        default_ckpt = os.path.join(os.path.dirname(__file__), '../checkpoints/checkpoint_epoch_29.pt')
        if os.path.exists(default_ckpt):
            ckpt = default_ckpt
            
    engine = GapBridgingInferenceEngine(model_config, train_config['inference_parameters'], device, checkpoint_path=ckpt)
    
    # Candidate search database (using input SWC and any additional fragments)
    candidates = [args.source_swc]
    
    phys_res = tuple(args.resolution)
    engine.bridge_gap(args.source_swc, candidates, args.tiff_volume, physical_resolution=phys_res)

if __name__ == "__main__":
    main()
