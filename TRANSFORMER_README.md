# NeuroGramLM: Tree-Aware Transformer

This document tracks the step-by-step implementation of the NeuroGramLM transformer architecture.

## Step 1: Vocabulary & Dataloader Setup (`dataset.py`)

### Overview
Before building the Transformer architecture, we must define the vocabulary mapping and prepare the data pipeline to handle massive 3D neuronal trees without encountering Out-Of-Memory (OOM) errors. 

### 1. Integer Vocabulary Mapping
Our multi-stream tokenized dataset uses string identifiers, which are mapped to integers for PyTorch `Embedding` layers:
- **Special Tokens** (`<PAD>`, `<START>`, `<END>`, `<BIF>`, `<END_BIF>`, `<POP>`, `<MASK_REG>`, `<JUMP>`): Indices `0` to `7`.
- **`<GEO_x>` Tokens**: Offset by `len(SPECIAL_TOKENS)`.
- **`<INV_x>` Tokens**: Offset similarly for the invariant stream.
- **`<REG_x>` Tokens**: Scaled up to 1500 to accommodate CCFv3 brain region identifiers. If a dataset lacks region info, the model uses `<MASK_REG>`.

### 2. Context Chunking (OOM Prevention)
A single SWC file can contain up to 50,000 nodes. Feeding this directly into a standard $O(N^2)$ Transformer will immediately cause an OOM leak on standard hardware (e.g., RTX 4090 16GB). 
To solve this, `NeuroGramDataset` implements a **sliding/chunked context window**:
- Sequences are chunked into a maximum length of `MAX_LEN = 2048`.
- If a fragment is shorter than 2048, it is padded with `<PAD>`.

### 3. Graph Laplacian Positional Encoding (Graphormer PE)
To ensure the Transformer understands the true 3D topology of the neuron (rather than just the 1D sequential DFS traversal), we use **Graph Laplacian Positional Encodings (PE)**.
- For each 2048-token chunk, the dataset reconstructs an undirected adjacency matrix `A` by tracing the `<BIF>` and `<POP>` tokens using a stack.
- It computes the normalized Graph Laplacian $L = I - D^{-1/2} A D^{-1/2}$.
- The first $k=8$ eigenvectors of $L$ are extracted.
- Because this computation scales at $O(N^3)$, chunking $N$ to 2048 allows it to run in milliseconds and consume virtually zero VRAM, completely bypassing the traditional memory bottlenecks of Graphormers.


## Step 2: Multi-Stream Transformer Architecture (`transformer_model.py`)

### Overview
The core architecture is an autoregressive / MLM transformer that predicts all three parallel streams. We employ separated embeddings and separate prediction heads.

### 1. Multi-Stream Embedding Layer
Instead of a single vocabulary, we sum embeddings from three streams:
E(token) = E_geo(GEO_x) + E_inv(INV_x) + E_reg(REG_x)
This allows the model to learn geometry and region semantics concurrently.

### 2. OOM Constraints on RTX 4090 (16GB)
Using the `measure_model_memory()` function, we verified that a 6-layer, 8-head Transformer with `d_model=512` utilizing the 2048 sliding context chunking fits gracefully within the 16GB VRAM limit.

## Step 3: Autoregressive Training Objective (`train.py`)

### Overview
The transformer learns to generate entire 3D neuronal trees from scratch using Autoregressive (Causal) Next-Token Prediction.

### 1. Multi-Stream Cross Entropy
Because each token consists of 3 parallel sub-tokens (INV, GEO, REG), the training loop computes three separate Cross-Entropy losses and sums them up.

### 2. Causal Masking
An upper-triangular causal attention mask (-inf) ensures the model only attends to past nodes in the DFS traversal, preventing data leakage during branch generation. We pad sequences dynamically and apply `src_key_padding_mask` to prevent attention to `<PAD>` tokens.

## Step 4: Unsupervised Skeletonization (`skeletonize_volume.py`)

### Overview
During test-time inference (when we do not have ground-truth SWC sequences but only raw intensity volumes), we must extract partial topological skeletons from the raw data.

### 1. Frangi Filter (Ridge Detection)
We use the Frangi Vesselness Filter (`skimage.filters.frangi`) to detect tube-like structures (neurites) by calculating the Hessian eigenvalues at each voxel in 3D.

### 2. 3D Skeletonization & Combined SWC Extraction
The top percentile of the vesselness map is thresholded into a binary mask. We then apply Lee’s 3D skeletonization (`skimage.morphology.skeletonize`) to thin the neurites into 1-voxel wide graphs. The script automatically reads XYZ spacing from `config.json`, clusters fragments, and scales spatial coordinates before extracting them into a single, topologically unified `extracted_fragments.swc` file.

## Step 5: Intensity-Guided Spline Routing (`inference_routing.py`)

### Overview
During autoregressive generation, the Transformer predicts the next spatial coordinate (GEO token) that connects two fragments. We use `skimage.graph.route_through_array` to trace the actual voxel-by-voxel path.

### Path of Least Resistance
The algorithm calculates the cheapest path between the end of Fragment A and the start of Fragment B across the intensity mountain ridge (the Frangi Vesselness map). The cost function is `1.0 / (vesselness + 1e-6)`. This guarantees that the generated linkages remain biologically plausible and follow the actual underlying intensity signals. It also supports **Atlas Space Routing** (without a raw volume), utilizing uniform Euclidean distance to route purely within abstract spatial bounds.

## Step 6: LM-Guided Topology Inference (`merge_fragments.py`)

### Overview
This is the final script that unifies the unsupervised topological fragments with the Language Model's neuron grammar. 

### Implementation
1. The script loads all disconnected topological fragments from `extracted_fragments.swc`.
2. The terminal (leaf) branches of these fragments are traced back to their root, tokenized, and fed into the `NeuroGramTransformer`.
3. The Transformer acts as a topological anchor generator, autoregressively predicting the optimal `GEO` spatial token for where this branch should continue.
4. The predicted anchor token is matched against the root nodes of all other unmerged fragments to find the most probable, biologically-grammatical connection.
5. The `inference_routing` engine is invoked to route a physical spline through the raw intensity volume linking the leaf to the predicted root.
6. The fully joined neuron graph is saved to `final_merged_neuron.swc`.

## Step 7: Evaluation & Metrics (`compare_swc.py`)

### Overview
After the transformer has reconstructed or merged SWCs, we must computationally validate the topological accuracy against ground-truth SWCs.

### Implementation
The `compare_swc.py` script parses both the original and reconstructed `.swc` files, and automatically prints quantitative comparison metrics for Node Count, Total Path Length in microns, and Bounding Box Extents (X,Y,Z).