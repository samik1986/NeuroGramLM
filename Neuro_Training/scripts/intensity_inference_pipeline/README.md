# NeuroGramLM: Intensity Inference Pipeline

The Intensity Inference Pipeline is an end-to-end biological structure extraction framework. It operates directly on raw unannotated volumetric TIFF images (intensity data) and produces fully contiguous, gap-bridged neuronal morphology trees (SWC format). 

This pipeline is uniquely designed for **unsupervised zero-shot extraction** combined with **generative gap-bridging inference**, allowing it to parse multi-billion voxel volumes without requiring supervised segmentation masks.

## Core Algorithmic Workflow

1. **GPU-Accelerated Soma Masking**:
   - Neuronal somas (cell bodies) are massive spherical structures that disrupt skeletonization (generating dense "hairballs"). 
   - We utilize a PyTorch-accelerated, multi-scale morphological approach. The volume is downsampled and convolved with a 3D spherical kernel based on physical resolution. 
   - A threshold requires 70% fill to capture even hollow/faint somas, which are then dilated and masked out (zeroed) from the volume.

2. **GPU-Accelerated Adaptive Local Thresholding (Faint & Thick Ridges)**:
   - A global threshold fails on thick neurons (flat mountain ridges) and faint paths in dark backgrounds.
   - We employ a **Massive Multi-Scale Background Estimator**: the chunk is downsampled 4x and pooled with a `15x15x15` kernel (effective 60-voxel/6.6μm receptive field).
   - Voxels are classified as ridges if they are merely **5% brighter** than this local massive background and above the absolute camera noise floor. This guarantees thick neurons remain solid (preventing hollowing) and faint neurons are extracted.

3. **Highly-Parallel Chunked 3D Skeletonization**:
   - Standard 3D topological skeletonization (`skimage`) takes ~45 minutes for a 2-Billion voxel array.
   - We utilize `joblib` to dynamically slice the binary volume into 36 overlapping chunks (e.g., `271x512x512` with 32-voxel overlap) and dispatch them across all available CPU cores (up to 128-core parallelization).
   - Skeletons are then seamlessly stitched together, reducing skeletonization time to seconds.

4. **BFS Graph Extraction**:
   - A fast O(N) Breadth-First Search (using `collections.deque`) traces the 26-connectivity of the 3D skeleton to extract valid parent-child node relationships (`pid`).
   - Output: `unsupervised_seeds.swc` (typically containing tens of thousands of highly accurate but disconnected fragment graphs).

5. **Generative Gap-Bridging Inference (BioTower)**:
   - The disconnected seed fragments are ingested into the `TokenizationPipeline`.
   - `GeometricEmbedding`, `TopologicalEmbedding`, and `BioEmbedding` 3D CNNs encode the morphological context of the endpoints.
   - An auto-regressive Transformer bridge engine connects broken fragments spanning up to 35.0 microns, forming continuous biological neuron trees (`final_smoothed_neurons.swc`).

---

## How to Use

Run the pipeline by pointing it to your raw TIFF volume and specifying the physical resolution (Z, Y, X in microns):

```bash
python Neuro_Training/scripts/intensity_inference_pipeline/intensity_pipeline.py \
    --tiff_volume <path_to_tif> \
    --resolution 0.1102 0.1102 0.5 \
    --bio_threshold 0.1
```

**Outputs**:
- `intensity_pipeline_outputs/masked_volume.tif`: Intermediate volume with somas removed.
- `intensity_pipeline_outputs/unsupervised_seeds.swc`: The raw extracted skeleton fragments before inference.
- `intensity_pipeline_outputs/final_smoothed_neurons.swc`: The final gap-bridged, fully connected neuron trees.

---

## Parameters and Tuning

If the output contains artifacts or misses neurons, you can tune the heuristic parameters directly in `intensity_pipeline.py`.

### 1. Soma Masking Tuning
*Located in `SomaMasker.get_masking_func()`*
- **`min_diameter_um`** (default: `10.0`): The physical diameter cutoff for somas.
  - *Implication*: Increasing this causes smaller somas to bypass the mask, resulting in dense skeleton "hairballs". Decreasing it too much may accidentally mask out thick dendrite intersections.
- **Core fill threshold (`k_sum * 0.70`)**: The density requirement for erosion.
  - *Implication*: Increasing to `0.90` demands perfectly solid, bright spheres (fails on hollow/faint somas). Decreasing to `0.50` will aggressively mask anything slightly spherical, risking the erasure of curved thick dendrites.

### 2. Adaptive Local Thresholding Tuning
*Located in `gpu_adaptive_threshold()`*
- **Local Background Multiplier (`local_bg * 1.05`)**: The relative contrast requirement.
  - *Implication*: Increasing to `1.15` will enforce strict, high-contrast ridges but will completely erase faint terminal paths. Decreasing to `1.01` will extract faint paths but also massively amplify background noise, connecting everything into a web.
- **Absolute Noise Floor (`1050.0`)**: The hard cutoff for camera noise.
  - *Implication*: If set too low, the 5% local contrast will pick up thermal camera noise as valid ridges. If set too high, faint neurons in completely dark regions will be clipped.
- **Receptive Field Pool Size (`kernel_size=15` on 4x downsample)**: The size of the local background estimate.
  - *Implication*: This defines the *maximum thickness* of a neuron that will remain a solid block. If you decrease this, thick neurons will dominate their own local background, causing them to be thresholded out in the center. This "hollowing" effect results in two parallel skeleton ridges instead of a single medial centerline. 

### 3. Gap-Bridging Inference Tuning
*Passed via CLI Arguments*
- **`--bio_threshold`** (default: `0.1`): The generative confidence threshold for bridging gaps.
  - *Implication*: Lowering this (e.g., `0.05`) encourages the model to bridge aggressive, distant gaps, which is great for sparse fragmentation but increases the risk of false merges (connecting two different neurons). Raising it forces the model to only bridge fragments with extremely high morphological and directional alignment.
