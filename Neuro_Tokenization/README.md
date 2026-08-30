# Neuro_Tokenization Framework

**Author**: Samik Banerjee  
**Date**: 2026-08-30  

Welcome to the `Neuro_Tokenization` framework. This package provides a modular, biologically meaningful, and mathematically robust tokenization strategy for neuronal morphologies (such as SWC files) in CCFv3 space.

## Features
- **Invariance**: Tokens are designed to be scale, rotation, and shift invariant, using intrinsic geometry rather than absolute coordinates.
- **Multimodal**: Tokens encapsulate spatial topology, intrinsic geometry (tortuosity, curvature), and contextual background data.
- **Fragment Stitching Ready**: Designed to handle unordered fragments with potential missing modalities (via Modality Masking).
- **GPU & Multicore Optimized**: Uses PyTorch for millisecond-scale geometric tensor vectorization on GPUs, and Python's ProcessPoolExecutor to fully saturate all CPU cores during file processing.
- **Modular Pipeline**: Built with step-by-step processing, tracking, and debugging.

## Documentation
- Read [ALGORITHM.md](./ALGORITHM.md) for a comprehensive mathematical explanation of the geometric features extracted and the rationale behind the invariance strategy.
- Modify parameters in `config.json` to tune smoothing, radii, and quality thresholds.

## Code Structure (Modular Pipeline)
The tokenization pipeline is broken down into four core modules:
1. `core/features.py`: Contains mathematical functions to compute Tortuosity, Curvature, Inertia Tensors, and Strahler Order.
2. `core/tokenizer.py`: The `MultimodalTokenizer` that aggregates features, handles the Depth-First Traversal of the tree, and applies Modality Masking for missing data.
3. `core/quality.py`: Evaluates a "Token Quality Score" to determine if a fragment is biologically meaningful.
4. `core/pipeline.py`: The execution engine that tracks the progress of each step, manages logging/debugging, and ensures smooth execution.

## Output Format & The Multimodal Grammar
The pipeline outputs JSON files (e.g., `_tokens.json`) representing the structural tokens of the neuron. 
The grammar output includes:
- **Continuous Embeddings**: Raw geometric properties.
- **Multimodal Codebook IDs (VQ)**: To convert raw continuous signals into a strict linguistic dictionary, the pipeline utilizes a factorized Multimodal Codebook. Instead of one monolithic codebook that breaks when data is missing, we use independent sub-codebooks for each modality (e.g., Tortuosity Codebook, Inertia Codebook). 
  - If a vocabulary model is present, the continuous values are mapped to these discrete cluster IDs (e.g., Tortuosity ID: 42, Curvature ID: 117). 
  - Missing modalities receive an attention mask of `0.0`, allowing the Language Model to effortlessly ignore missing data across partial fragments.

### Hardware & Optimization Assumptions
- **PyTorch GPU Acceleration**: Geometric calculations (Tortuosity, Curvature, Inertia) are automatically vectorized using PyTorch (`torch.cdist`, `torch.cross`). This brings extraction time for massive fragments down to single-digit milliseconds. If CUDA is not detected, it smoothly falls back to a NumPy implementation.
- **Multiprocessing Tokenization**: Processing 180k+ SWC files is heavily I/O and JSON bound. The pipeline automatically leverages `concurrent.futures.ProcessPoolExecutor` to distribute the load across all available CPU cores.
- **VQ Vocabulary Scaling**: The K-Means Codebook is strictly capped to train on a uniform random sample of 500 SWC files. Because every SWC file is an entire morphological tree, 500 files actually yield over 1 million continuous data points. For a cluster size of 512, this guarantees >2,000 samples per centroid, making underfitting mathematically impossible while saving massive amounts of VQ initialization time.

## Walkthrough: How to Use

Here is a basic example of how to process an SWC fragment through the pipeline:

```python
from core.pipeline import TokenizationPipeline
import json

# 1. Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# 2. Initialize the modular pipeline
pipeline = TokenizationPipeline(config)

# 3. Process an SWC file (or a fragment)
# This will run the parser -> feature extraction -> tokenization -> quality check
result = pipeline.process_fragment('path/to/neuron_fragment.swc')

if result['success']:
    tokens = result['tokens']
    quality_score = result['quality_score']
    print(f"Generated {len(tokens)} tokens with biological quality score: {quality_score:.2f}")
    
    # You can append new modalities easily because of the modular structure
    # e.g., if you have background intensity:
    # tokens = pipeline.append_modality(tokens, 'background_intensity', [0.5, 0.8, ...])
else:
    print("Pipeline failed:", result['error_log'])
```

## Adding New Modalities
The tokenizer is extensible. To add a new property (e.g., 'myelin_thickness'):
1. Add the extraction logic in `core/features.py`.
2. Register the modality key in the `MultimodalTokenizer` inside `core/tokenizer.py`.
3. If the modality is missing at test time, the tokenizer will automatically substitute it with the `mask_modality_value` defined in `config.json`.
