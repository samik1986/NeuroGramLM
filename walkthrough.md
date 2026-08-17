# NeuroVLM Tree-Aware Transformer Walkthrough

We have successfully implemented the complete pipeline for the **NeuroGram Multi-Stream Transformer** to parse, process, and merge neuronal fragments from arbitrary 3D volumes.

## Architecture Highlights
- **Multi-Stream Embedding**: A `MultiStreamEmbedding` layer fuses parallel `GEO`, `INV`, and `REG` streams into a unified feature space.
- **Topological Encoding**: To handle the tree structures, we implemented **Graph Laplacian Positional Encoding**. This encodes the structural topology of the neurons into the context window, bypassing standard 1D sine/cosine positional encodings.
- **Autoregressive Masking**: We added causal attention masking (`generate_square_subsequent_mask`) to allow the Transformer Encoder to autoregressively predict spatial offsets sequentially, preventing data leakage during training.
- **OOM Prevention**: 
    - Analyzed the memory constraints for an RTX 4090 (16GB VRAM target).
    - Reduced $O(N^2)$ memory scaling by chunking long neurons into overlapping sequences of `2048` tokens using the `NeuroGramDataset` dataloader. 
    - Confirmed parameters total ~21.5 Million, safely fitting the memory limit.

## Unsupervised Generation from Raw Volumes
During inference on entirely raw volumes:
1. **Ridge Detection**: We use `skimage.filters.frangi` to isolate the 3D neurite topologies based on Hessian eigenvalues without any supervised training labels.
2. **Skeletonization**: `skimage.morphology.skeletonize` thins these topological features into 1-voxel wide graphs (fragments).
3. **Intensity-Guided Routing**: We combined the Transformer's spatial prediction capabilities with Dijkstra's algorithm (`skimage.graph.route_through_array`) to route connections between disconnected fragments along the actual intensity mountains of the raw data.

All steps have been fully documented in the newly created [TRANSFORMER_README.md](file:///c:/Users/banerjee/Desktop/Current%20Work/NeuroVLM/TRANSFORMER_README.md) for future reference and reproducibility. We verified both forward/backward gradient flows (dummy testing) and OOM footprint tracking sequentially for each step.
