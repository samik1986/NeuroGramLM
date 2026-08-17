# NeuroGramLM: End-to-End Pipeline Runbook

This guide provides step-by-step instructions for running the complete NeuroGramLM pipeline: from preparing the ground-truth training dataset and training the transformer, to running unsupervised inference on raw volumetric data, and finally merging topological fragments into biologically meaningful neuronal SWCs.

## Prerequisites
1. Ensure your hardware supports CUDA (The transformer is optimized to fit within a 16GB VRAM limit using 2048-token context windows).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your pipeline parameters (e.g. data paths, physical XYZ spacing, vocabulary sizes) in `config.json`.

---

## Phase 1: Training the Language Model

### Step 1: Tokenize Ground-Truth SWCs
Convert your ground-truth neuronal trees (`.swc` files) into a sequence of discrete tokens (Geometry, Invariant, and Region) via Vector Quantization.
```bash
python tokenize_swc_vq.py
```
- **Input**: Directory of `.swc` files.
- **Output**: `dummy_dataset.jsonl` (and fitted KMeans models `geo_kmeans.pkl`, `inv_kmeans.pkl`).

### Step 2: Train the NeuroGram Transformer
Train the multi-stream causal transformer on the tokenized dataset. The dataloader automatically chunks the neuronal sequences and computes Graph Laplacian Positional Encodings to map 3D topology without causing OOM memory leaks.
```bash
python train.py
```
- **Input**: `dummy_dataset.jsonl`
- **Output**: Model checkpoints containing the learned biological grammar of how neurons branch and connect.

---

## Phase 2: Inference on Raw Volumes

When you have a novel raw TIFF volume without ground-truth skeletons, follow these steps to extract and connect fragments using the trained model.

### Step 3: Unsupervised Topological Extraction
Extract distinct neurite fragments directly from the raw intensity volume using the Frangi Vesselness Filter and Lee's 3D skeletonization.
```bash
python skeletonize_volume.py
```
- **Input**: Raw TIFF volume (configured in `config.json`).
- **Output**: `extracted_fragments.swc` (contains all disconnected topological graphs dynamically scaled to true physical micron space).

### Step 4: Language-Model Guided Merging
Use the trained Transformer to autoregressively predict missing connections between the fragments. The model generates topological anchors, and Dijkstra's algorithm routes physical splines along the underlying raw intensity ridges (or uniformly in Atlas Space) to connect them.
```bash
python merge_fragments.py
```
- **Input**: `extracted_fragments.swc` and the Vesselness map.
- **Output**: `final_merged_neuron.swc` (the final, biologically meaningful, fully-connected neuronal tree).

---

## Phase 3: Evaluation & Metrics

To validate the model's performance and ensure the Vector Quantization preserves biological topology, you can compare the reconstructed/merged SWCs against the ground truth.

### Topology Comparison (`compare_swc.py`)
This script compares original `.swc` files against the generated ones and prints out topological comparison metrics (Node Count, Total Path Length in microns, and Bounding Box Extent).
```bash
python compare_swc.py
```
- **Input**: Original `.swc` files (e.g., in `SWCs/`) and reconstructed `detokenized_*.swc` or merged files.
- **Output**: Terminal printout of topology comparison metrics.

---

## Configuration Reference (`config.json`)
All parameters are centralized in `config.json` for easy experiment tracking.
- `input_output.volume_path`: Path to the raw TIFF volume (e.g. your channel 3 multi-channel TIFF).
- `input_output.spacing_xyz`: The physical resolution of your voxels (e.g., `[0.11, 0.11, 0.5]`).
- `tokenization.max_context_length`: The chunk size (2048) to prevent OOM errors.
- `transformer`: Model architecture dimensions (layers, heads, d_model).
- `unsupervised_detection.frangi_sigmas`: Scales for the 3D ridge detection filter.

---

## Hardware Notes
- **OOM Prevention**: The NeuroGram transformer incorporates sliding 2048-token context windows and memory-efficient Graph Laplacian positional encodings ($k=8$).
- **Target GPU**: The model guarantees an execution footprint of ~8.1 GB VRAM for a batch size of 8, allowing the full pipeline to run efficiently on a 16GB GPU (like the RTX 4090) without Out-of-Memory crashes.
