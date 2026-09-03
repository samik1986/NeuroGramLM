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

## How to Run

Here is the quickstart guide for executing the pipelines from the command line. Ensure your environments are activated and you are in the repository root.

### Training the Model
To start a fresh training run using the defined hyperparameters:
```bash
cd Neuro_Training
python scripts/train.py
```

### Running Inference (Gap Bridging)
To bridge broken gaps in raw SWC files utilizing the Zero-Shot Bio Tower (TIFF intensity crops):
```bash
cd Neuro_Training
python scripts/inference.py
```

### Incremental Retraining (New Data)
To validate new SWCs that might not be in CCF space, tokenize them, and incrementally update the model:
```bash
cd Neuro_Retraining
python scripts/run_incremental.py
```
