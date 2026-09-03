# NeuroGramLM

NeuroGramLM is a Large Language Model designed specifically for the morphological and topological representation of whole-brain biological neurons. It treats the complex 3D branching structures of neurons as a specialized "language," enabling high-accuracy reconstruction, gap bridging, and analysis of neuroanatomy.

## Project Architecture

This repository is strictly modularized into four isolated pipelines. Each module contains its own documentation, configurations, and algorithm details.

### 1. `Neuro_Tokenization/`
The foundational layer. This module takes raw `.swc` graph files and converts physical traits (curvature, tortuosity, branching angles, Strahler order) into discrete tokens via Vector Quantization (VQ). This transforms continuous 3D graphs into text-like sequences.

### 2. `Neuro_Model/`
The core PyTorch architecture.
- A **Multi-Tower Encoder** processes geometric shape and topological hierarchy independently before fusing them into a shared latent space.
- A **Zero-Shot 3D CNN (Bio Tower)** allows the model to "see" raw microscopy intensity images at test time without requiring them during training.
- A **Multi-Tower Decoder** autoregressively generates missing neuron segments.

### 3. `Neuro_Training/`
The main training and inference engine.
- Contains the robust teacher-forced training loop.
- Features a **Two-Stage Gap Bridging Inference Engine** that uses fast latent KNN search combined with the zero-shot Bio Tower to connect broken fragments across massive brain volumes with high accuracy.

### 4. `Neuro_Retraining/`
The continuous/incremental learning module.
- Designed to ingest *new* SWC datasets that are not explicitly in the Allen CCFv3 coordinate space.
- Employs a strict **Biological Plausibility Gating Mechanism** to ensure that new data, despite having arbitrary scaling or origin points, conforms perfectly to the relative biological distributions of CCFv3 neurons before allowing the model to update its weights.

---

## How to Run (Unified CLI)

We have implemented a master orchestrator script `neurogram.py` at the root directory. You no longer need to `cd` into individual submodules. Instead, run the desired pipeline step directly from the root using the `--step` argument.

### 1. Tokenize Raw SWCs
```bash
python neurogram.py --step tokenize
```
*(Optionally override the input dir: `--input_dir data/my_swcs`)*

### 2. Train the Model from Scratch
```bash
python neurogram.py --step train
```
*(Optionally override epochs: `--epochs 150`)*

### 3. Run Inference (Gap Bridging)
```bash
python neurogram.py --step infer --source_swc data/raw/frag_001.swc --tiff_volume data/raw/brain.tif
```

### 4. Incremental Retraining (New Data)
Validate new, arbitrary-scaled SWCs against the CCFv3 latent space and resume training on them:
```bash
python neurogram.py --step retrain --checkpoint checkpoints/latest.pt
```
