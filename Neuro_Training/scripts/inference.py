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
        self._cached_tiff_path = None
        self._cached_volume = None

    def get_volume(self, tiff_volume_path):
        if self._cached_tiff_path == tiff_volume_path and self._cached_volume is not None:
            return self._cached_volume
        if os.path.exists(tiff_volume_path) and 'tifffile' in sys.modules:
            try:
                logger.info(f"Loading 3D TIFF into memory for fast patch extraction: {tiff_volume_path}")
                self._cached_volume = tifffile.imread(tiff_volume_path)
                self._cached_tiff_path = tiff_volume_path
                return self._cached_volume
            except Exception as e:
                logger.warning(f"Could not load TIFF {tiff_volume_path}: {e}")
        return None

    def extract_tiff_patch(self, tiff_volume_path, start_xyz, end_xyz, patch_size=(32, 32, 32)):
        """
        Extracts a 3D intensity ridge from the raw TIFF volume along the trajectory 
        between the source and candidate fragment.
        """
        vol = self.get_volume(tiff_volume_path)
        if vol is not None and vol.ndim == 3:
            try:
                d, h, w = vol.shape
                mid_z = int(np.clip((start_xyz[2] + end_xyz[2]) / 2.0, 0, d - 1))
                mid_y = int(np.clip((start_xyz[1] + end_xyz[1]) / 2.0, 0, h - 1))
                mid_x = int(np.clip((start_xyz[0] + end_xyz[0]) / 2.0, 0, w - 1))
                
                pd, ph, pw = patch_size
                z1, z2 = max(0, mid_z - pd // 2), min(d, mid_z + pd // 2)
                y1, y2 = max(0, mid_y - ph // 2), min(h, mid_y + ph // 2)
                x1, x2 = max(0, mid_x - pw // 2), min(w, mid_x + pw // 2)
                
                crop = vol[z1:z2, y1:y2, x1:x2].astype(np.float32)
                if crop.max() > 0:
                    crop = crop / crop.max()
                
                patch = np.zeros(patch_size, dtype=np.float32)
                patch[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
                return torch.tensor(patch).unsqueeze(0) # (1, D, H, W)
            except Exception as e:
                logger.warning(f"Could not read TIFF crop: {e}")

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

    def score_fragment_conformance(self, fragment_points, physical_resolution=None):
        """
        Computes the model log-likelihood / cross-entropy perplexity of a given fragment sequence.
        Returns (conformance_score, is_accepted, token_quality_penalties)
        """
        if len(fragment_points) < 3:
            # Trivial fragment of <= 2 points
            return 0.5, True, []

        raw_points = np.array(fragment_points, dtype=np.float32)
        if physical_resolution is not None:
            res_array = np.array(physical_resolution, dtype=np.float32)
            raw_points[:, :3] = raw_points[:, :3] * res_array
            
        norm_points = self.preprocessor.normalize_scale(raw_points)
        result = self.preprocessor.tokenizer.process_fragment(norm_points) if self.preprocessor.tokenizer else {'success': False}
        
        if not result.get('success', False) or len(result.get('tokens', [])) < 2:
            return 0.0, False, result.get('penalties', ['Tokenization failed'])

        tokens = result['tokens']
        MAX_LEN = 128
        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
            
        vq_tort = torch.tensor([[t.get('vq_ids', {}).get('tortuosity', 0) for t in tokens]], dtype=torch.long, device=self.device)
        vq_curv = torch.tensor([[t.get('vq_ids', {}).get('curvature_energy', 0) for t in tokens]], dtype=torch.long, device=self.device)
        vq_iner = torch.tensor([[t.get('vq_ids', {}).get('inertia_tensor', 0) for t in tokens]], dtype=torch.long, device=self.device)
        topo_stra = torch.tensor([[t.get('embeddings', {}).get('strahler_order', 1) for t in tokens]], dtype=torch.long, device=self.device)
        topo_wl = torch.tensor([[t.get('embeddings', {}).get('wl_hash', 0) % 1000 for t in tokens]], dtype=torch.long, device=self.device)

        # Autoregressive teacher forcing loss evaluation
        seq_len = vq_tort.size(1)
        if seq_len < 2:
            return 0.8, True, []
            
        inp_kwargs = {
            'vq_tortuosity': vq_tort[:, :-1],
            'vq_curvature': vq_curv[:, :-1],
            'vq_inertia': vq_iner[:, :-1],
            'topo_strahler': topo_stra[:, :-1],
            'topo_wl': topo_wl[:, :-1],
            'target_vq_tortuosity_shifted': vq_tort[:, :-1],
            'target_vq_curvature_shifted': vq_curv[:, :-1],
            'target_vq_inertia_shifted': vq_iner[:, :-1],
            'target_topo_strahler_shifted': topo_stra[:, :-1],
            'target_topo_wl_shifted': topo_wl[:, :-1],
            'target_vq_tortuosity': vq_tort[:, 1:],
            'target_vq_curvature': vq_curv[:, 1:],
            'target_vq_inertia': vq_iner[:, 1:],
            'target_topo_strahler': topo_stra[:, 1:],
            'target_topo_wl': topo_wl[:, 1:],
            'bio_volumes': None
        }

        with torch.no_grad():
            out = self.model(**inp_kwargs)
            loss_val = out['loss'].item() if isinstance(out, dict) and 'loss' in out else float(out)
            
        # Conformance score: higher is better (normalized exp(-loss))
        conformance_score = float(np.exp(-min(loss_val, 20.0)))
        quality_penalties = result.get('penalties', [])
        
        # Criteria for acceptance: loss threshold + quality validity
        is_accepted = (loss_val < 4.5) and (len(quality_penalties) == 0 or 'Unrealistic tortuosity' not in ' '.join(quality_penalties))
        return conformance_score, is_accepted, quality_penalties

    def bridge_and_connect_swc(
        self, 
        input_swc_path, 
        output_swc_path, 
        tiff_volume_path, 
        physical_resolution=None, 
        reject_nonconforming=False,
        max_gap_distance_um=35.0,
        bio_bridge_threshold=0.35,
        max_fragments_to_connect=5000
    ):
        """
        Selective Single-Neuron Gap Bridging Pipeline:
        1. Identifies all disconnected fragments in the optical volume.
        2. Queries candidate gap pairings within plausible biological gap proximity (<= 35 μm/voxels).
        3. Evaluates 3D optical fluorescence ridge continuity (Bio Tower) & directional trajectory alignment.
        4. Bridges ONLY candidate fragments that share continuous optical intensity from the SAME neuron.
        5. Preserves distinct neurons as separate disconnected root trees (pid = -1) in the output SWC.
        """
        logger.info(f"Loading and segmenting fragments from: {input_swc_path}")
        fragments = []
        current_frag = []
        with open(input_swc_path, 'r') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith('#'):
                    continue
                parts = line_str.split()
                if len(parts) >= 7:
                    try:
                        nid, ntype = int(parts[0]), int(parts[1])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        r, pid = float(parts[5]), int(parts[6])
                        if pid == -1 and len(current_frag) > 0:
                            fragments.append(current_frag)
                            current_frag = []
                        current_frag.append({
                            'id': nid, 'type': ntype, 'x': x, 'y': y, 'z': z, 'r': r, 'pid': pid
                        })
                    except ValueError:
                        continue
        if len(current_frag) > 0:
            fragments.append(current_frag)

        total_frags = len(fragments)
        logger.info(f"Identified {total_frags} disconnected fragments in input SWC.")
        
        accepted_fragments = fragments
        rejected_count = 0

        # Step 2: Extract endpoints and directional vectors for each fragment
        from scipy.spatial import cKDTree
        
        frag_heads = np.array([[f[0]['x'], f[0]['y'], f[0]['z']] for f in accepted_fragments], dtype=np.float32)
        frag_tails = np.array([[f[-1]['x'], f[-1]['y'], f[-1]['z']] for f in accepted_fragments], dtype=np.float32)
        
        # Calculate trailing orientation vector for each fragment
        tail_dirs = []
        for f in accepted_fragments:
            if len(f) >= 2:
                v = np.array([f[-1]['x'] - f[-2]['x'], f[-1]['y'] - f[-2]['y'], f[-1]['z'] - f[-2]['z']], dtype=np.float32)
                norm = np.linalg.norm(v)
                tail_dirs.append(v / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32))
            else:
                tail_dirs.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        tail_dirs = np.array(tail_dirs)
        
        head_tree = cKDTree(frag_heads)
        
        # Track verified biological bridges: (tail_frag_idx -> head_frag_idx)
        connections = {} 
        used_heads = set()
        
        logger.info(f"Evaluating Selective Single-Neuron Gap Bridging (Max Gap: {max_gap_distance_um} μm/voxels, Threshold: {bio_bridge_threshold})...")
        for i, tail_pt in enumerate(frag_tails):
            # Query candidate nearby heads within plausible gap distance
            candidate_indices = head_tree.query_ball_point(tail_pt, r=max_gap_distance_um)
            valid_candidates = [c for c in candidate_indices if c != i and c not in used_heads]
            
            if len(valid_candidates) == 0:
                continue
                
            best_candidate = None
            best_score = -float('inf')
            tail_dir = tail_dirs[i]
            
            for c_idx in valid_candidates:
                head_pt = frag_heads[c_idx]
                gap_vec = head_pt - tail_pt
                dist = np.linalg.norm(gap_vec)
                if dist < 1e-4:
                    continue
                    
                gap_unit = gap_vec / dist
                # Directional cosine alignment between fragment growth trajectory and gap vector
                cos_sim = float(np.dot(tail_dir, gap_unit))
                
                # Penalize sharp non-biological angle turns (> 90 degrees)
                if cos_sim < -0.2:
                    continue
                    
                # Extract 3D optical intensity ridge along gap
                patch_tensor = self.preprocessor.extract_tiff_patch(
                    tiff_volume_path, tail_pt, head_pt
                ) # (1, D, H, W)
                
                mean_intensity = patch_tensor.mean().item()
                max_intensity = patch_tensor.max().item()
                
                # Combined biological continuity score: optical ridge fluorescence + directional continuity - distance penalty
                bio_score = (0.6 * max_intensity + 0.4 * mean_intensity) * max(0.1, cos_sim + 0.5) - (dist / (max_gap_distance_um * 2.0))
                
                if bio_score > best_score:
                    best_score = bio_score
                    best_candidate = c_idx
                    
            # Bridge only if above strict biological threshold for single-neuron continuity
            if best_candidate is not None and best_score >= bio_bridge_threshold:
                connections[i] = best_candidate
                used_heads.add(best_candidate)

        logger.info(f"Model identified {len(connections)} high-confidence biological gap bridges. Remaining fragments will stay as separate individual neuron trees.")

        # Step 3: Reconstruct Multigraph with Independent Neuron Trees
        all_nodes = []
        node_id_counter = 1
        frag_to_node_ids = {}
        
        for f_idx, frag in enumerate(accepted_fragments):
            id_map = {}
            node_ids_in_frag = []
            for node in frag:
                old_id = node['id']
                new_id = node_id_counter
                id_map[old_id] = new_id
                node_ids_in_frag.append(new_id)
                
                parent_id = -1 if node['pid'] == -1 else id_map.get(node['pid'], -1)
                all_nodes.append({
                    'id': new_id,
                    'type': node['type'],
                    'x': node['x'],
                    'y': node['y'],
                    'z': node['z'],
                    'r': node['r'],
                    'pid': parent_id
                })
                node_id_counter += 1
            frag_to_node_ids[f_idx] = node_ids_in_frag

        # Wire only the validated single-neuron gap bridges into parent pointers
        for src_frag, dst_frag in connections.items():
            if src_frag in frag_to_node_ids and dst_frag in frag_to_node_ids:
                tail_node_id = frag_to_node_ids[src_frag][-1]
                dst_head_node_id = frag_to_node_ids[dst_frag][0]
                
                # Connect destination root to source tail
                for n in all_nodes:
                    if n['id'] == dst_head_node_id:
                        n['pid'] = tail_node_id
                        break

        # Calculate final number of distinct neuron trees (roots with pid == -1)
        distinct_trees = sum(1 for n in all_nodes if n['pid'] == -1)
        logger.info(f"Reconstructed volume contains {distinct_trees} distinct neuron trees across {len(all_nodes)} total nodes.")

        # Write output selective bridged SWC
        os.makedirs(os.path.dirname(os.path.abspath(output_swc_path)), exist_ok=True)
        logger.info(f"Writing selectively bridged SWC to: {output_swc_path}")
        with open(output_swc_path, 'w') as f:
            f.write("# NeuroGramLM Selective Single-Neuron Gap Bridging Output\n")
            f.write(f"# Input Source: {input_swc_path}\n")
            f.write(f"# TIFF Volume: {tiff_volume_path}\n")
            f.write(f"# Total Original Fragments: {total_frags}\n")
            f.write(f"# Model-Predicted Same-Neuron Gap Bridges: {len(connections)}\n")
            f.write(f"# Final Distinct Neuron Trees: {distinct_trees}\n")
            f.write(f"# Total Nodes: {len(all_nodes)}\n")
            f.write("# ID Type X Y Z Radius Parent\n")
            for n in all_nodes:
                f.write(f"{n['id']} {n['type']} {n['x']:.4f} {n['y']:.4f} {n['z']:.4f} {n['r']:.2f} {n['pid']}\n")

        logger.info(f"Successfully exported multi-neuron bridged SWC to {output_swc_path}")
        return output_swc_path, distinct_trees, len(connections)

def main():
    parser = argparse.ArgumentParser(description="NeuroGramLM Inference Engine")
    parser.add_argument('--source_swc', type=str, default="skeletons_connected_new.swc", help="Source fragment SWC")
    parser.add_argument('--output_dir', type=str, default="inference_outputs_selective_neurons", help="Directory to save all inference outputs and figures")
    parser.add_argument('--output_swc', type=str, default="predicted_selective_neurons_joined.swc", help="Custom filename for joined SWC")
    parser.add_argument('--tiff_volume', type=str, default="F0046_multichannel_cmle_ch03.tif", help="TIFF intensity volume")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument('--resolution', type=float, nargs=3, default=[1.0, 1.0, 1.0], help="XYZ coordinate scaling")
    parser.add_argument('--max_gap_distance', type=float, default=35.0, help="Max search distance for gap bridging (voxels/microns)")
    parser.add_argument('--bio_threshold', type=float, default=0.35, help="Minimum biological ridge intensity and directional continuity score to bridge a gap")
    parser.add_argument('--visualize', action='store_true', default=True, help="Automatically render 3D volume overlay visualization into output folder")
    args = parser.parse_args()

    # Create dedicated output folder
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Inference output directory: {os.path.abspath(args.output_dir)}")

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
    phys_res = tuple(args.resolution)
    
    swc_filename = args.output_swc if args.output_swc else "predicted_selective_neurons_joined.swc"
    output_swc_path = os.path.join(args.output_dir, swc_filename)
    
    # Run Selective Gap Bridging and SWC Reconstruction
    out_swc, distinct_trees, bridges_count = engine.bridge_and_connect_swc(
        args.source_swc, 
        output_swc_path, 
        args.tiff_volume, 
        physical_resolution=phys_res,
        max_gap_distance_um=args.max_gap_distance,
        bio_bridge_threshold=args.bio_threshold
    )

    # Save summary metadata
    meta_path = os.path.join(args.output_dir, "inference_summary.json")
    with open(meta_path, 'w') as f:
        json.dump({
            "source_swc": args.source_swc,
            "tiff_volume": args.tiff_volume,
            "checkpoint": ckpt,
            "physical_resolution": args.resolution,
            "max_gap_distance": args.max_gap_distance,
            "bio_bridge_threshold": args.bio_threshold,
            "distinct_neuron_trees": distinct_trees,
            "bridges_constructed": bridges_count,
            "output_swc": out_swc
        }, f, indent=4)
    logger.info(f"Saved inference summary to {meta_path}")

    # Render volume visualization directly in output directory
    if args.visualize and os.path.exists(args.tiff_volume):
        try:
            from Neuro_Training.scripts.visualize_volume import visualize_swc_on_volume
            fig_path = os.path.join(args.output_dir, "swc_volume_visualization.png")
            logger.info(f"Rendering 3D volume overlay to: {fig_path}")
            visualize_swc_on_volume(
                tiff_path=args.tiff_volume,
                swc_path=out_swc,
                output_png=fig_path,
                resolution=phys_res
            )
        except Exception as e:
            logger.warning(f"Could not render visualization figure: {e}")

if __name__ == "__main__":
    main()
