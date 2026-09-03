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
- **Input Formats:** Accepts raw `.swc` files for neuron fragments and raw `.tif` (TIFF) files for volumetric microscopy intensity data. The inference engine also accepts a `(x, y, z)` physical resolution mapping (microns per voxel) via the `--resolution` argument.
- **Scale Normalization:** Because test data might be captured at an arbitrary micron scale compared to CCFv3, the inference preprocessor automatically applies any provided physical resolution scaling (`x * res_x, y * res_y, z * res_z`). It then normalizes the spatial bounding box of the SWC. This prevents continuous metrics (like inertia or curvature energy) from exploding outside the bounds of the VQ codebooks.
- **Rotation Invariance:** No spatial alignment/registration is required before tokenization. Because the geometric tokenization pipeline relies exclusively on relative intrinsic features (branching angles, tortuosity, scalar curvature), the model is inherently translation and rotation invariant.

## Usage

### 1. Training from Scratch
To start a new training run using the parameters defined in `config.json`:
```bash
cd Neuro_Training
python scripts/train.py
```
This will automatically save checkpoints inside the `checkpoints/` directory.

### 2. Incremental / Continual Training
If you acquire a new batch of CCFv3 tokenized data and want to continue training from an existing model (fine-tuning or incremental learning), use the `--resume_from` flag:
```bash
cd Neuro_Training
python scripts/train.py --resume_from checkpoints/checkpoint_epoch_50.pt
```
The script will load the saved model state, retain the learned embeddings, and continue iterating over your new dataloader.

### 3. Running Gap Bridging Inference
The inference script operates on raw biological SWC tracings and TIFF imaging files. You should supply the physical resolution to ensure precise mapping.
```bash
cd Neuro_Training
python scripts/inference.py --source_swc "data/raw/frag_001.swc" --tiff_volume "data/raw/brain.tif" --resolution 1.0 1.0 3.0
```
*Note: Ensure you update the mock file paths at the bottom of `inference.py` to point to your actual source SWC, candidate SWCs directory, and TIFF volume file before executing.*
