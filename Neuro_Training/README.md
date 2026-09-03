# NeuroGramLM Training & Inference

This module contains the logic for training the `Neuro_Model` Encoder-Decoder Transformer and running the Two-Stage Gap Bridging Inference engine.

## Directory Structure
- `config.json`: Hyperparameters for training loops and KNN/Bio thresholds for inference.
- `scripts/train.py`: The main PyTorch training loop with gradient clipping and modular loss handling.
- `scripts/inference.py`: Implements the 2-Stage gap bridging algorithm (Latent KNN Search -> Zero-Shot Bio Validation). It includes an **Inference Preprocessor** to handle raw SWC and TIFF inputs dynamically.
- `analysis/loss_graph.py`: Utility to parse JSONL training logs and generate matplotlib loss trajectories for all sub-towers.
- `checkpoints/`: Model weights are saved here.

## Overview
1. **Training:** 
   The model is trained on continuous neuron fragments using autoregressive teacher-forcing. Modality Dropout is applied so the model learns to bridge gaps relying purely on the Geometric and Topological encoders.
2. **Inference (Bridging):**
   When a gap is encountered, `inference.py` predicts the missing latent node, searches the database for candidate fragments using KNN, and then validates the top candidates by passing their real 3D intensity ridges through the Zero-Shot Bio Tower.

## Inference Data Assumptions
Unlike training, which uses pre-computed tokens from a CCFv3 dataset, the **inference engine operates on raw biological data:**
- **Input Formats:** Accepts raw `.swc` files for neuron fragments and raw `.tif` (TIFF) files for volumetric microscopy intensity data. The inference engine also accepts an optional `(x, y, z)` physical resolution mapping.
- **Scale Normalization:** Because test data might be captured at an arbitrary micron scale compared to CCFv3, the inference preprocessor automatically applies any provided physical resolution scaling. It then normalizes the spatial bounding box of the SWC. This prevents continuous metrics (like inertia or curvature energy) from exploding outside the bounds of the VQ codebooks.
- **Rotation Invariance:** No spatial alignment/registration is required before tokenization. Because the geometric tokenization pipeline relies exclusively on relative intrinsic features (branching angles, tortuosity, scalar curvature), the model is inherently translation and rotation invariant.
