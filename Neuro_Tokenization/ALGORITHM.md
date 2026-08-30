# Algorithm and Mathematical Foundations

**Author**: Samik Banerjee  
**Date**: 2026-08-30  

This document explains the mathematical foundations of the `Neuro_Tokenization` framework, ensuring that the multimodal tokens are rotation, scale, and shift invariant, and that they capture biologically meaningful structures.

## 0. Multimodal Vector Quantization (VQ) & Vocabulary

To construct a robust linguistic grammar from geometric continuous features, we apply **Multimodal Vector Quantization (VQ)**. This maps the infinite continuous space into a finite, discrete vocabulary (a codebook).

### Logic & Anti-Overfitting Strategy
Neuronal morphologies can vary wildly between regions. If a vocabulary is learned on a biased subset (e.g., only densely branched cortical neurons), it will severely underfit long, sparse projection neurons.
To prevent overfit/underfit:
1. **Stratified Random Sampling**: We traverse all folders in the `../SWCs` directory and uniformly sample fragments. This ensures the VQ codebook sees diverse morphologies without allowing any specific folder to dominate.
### The Multimodal Codebook Strategy
To convert raw geometric shapes into a linguistic vocabulary, we utilize a **Multimodal Codebook**.
A traditional VQ-VAE uses a single codebook. However, because our neuronal fragments often have missing data (e.g., missing branching angles, incomplete soma topologies), a monolithic codebook fails when any single modality is absent.

Instead, our Multimodal Codebook is **factorized**:
We maintain independent vector sub-codebooks $\mathbb{R}^{K \times D}$ for each distinct modality (e.g., $K=256$ for Tortuosity, $K=512$ for Inertia). 
During tokenization, the continuous embeddings of a fragment are quantized independently by each sub-codebook. The final unified token is a multimodal tuple of discrete IDs. This structure allows the Language Model to seamlessly mask out missing modalities (using the attention mask) while perfectly interpreting the modalities that *are* present.

### Example Token
A raw continuous token might extract a tortuosity of `1.12` and a curvature energy of `0.45`. 
During Vector Quantization:
- `1.12` maps to the nearest centroid in the Tortuosity Codebook (e.g., ID `42`).
- `0.45` maps to the nearest centroid in the Curvature Codebook (e.g., ID `117`).
The resulting discrete multimodal tuple is `[42, 117, ...]`, which perfectly mimics the input format expected by a multimodal Transformer.

## 1. Achieving Invariance
Global spatial coordinates $(x,y,z)$ are highly dependent on the bounding box and orientation of the scan. To achieve invariance:
- **Shift Invariance**: We use relative displacements $\Delta \vec{p}_i = \vec{p}_i - \vec{p}_{i-1}$ or compute features independent of the origin.
- **Rotation Invariance**: Features are based on scalar products (like lengths, angles) or local frames (Frenet-Serret) which rotate with the structure.
- **Scale Invariance**: Lengths are normalized against a bounding box area or total path length.

## 2. Morphological Features (Intrinsic Geometry)

### 2.1 Tortuosity
**Concept**: Measures how "twisted" a neuronal branch is.
**Formula**: 
$$ T = \frac{L_{arc}}{L_{chord}} $$
Where $L_{arc}$ is the actual path length of the segment along the curve, and $L_{chord}$ is the straight-line distance between its endpoints.
**Parameter**: `tortuosity_window_size` (default 5). We compute this locally over a sliding window of nodes to capture micro-tortuosity, ensuring we don't average out small wiggles.

### 2.2 Curvature Energy and Frequency
**Concept**: Curvature $\kappa$ measures the rate of change of the tangent vector. Energy represents the total bending effort.
**Formula**: 
Let $\vec{r}(s)$ be the parameterized curve by arc length. The curvature is $\kappa = \left\| \frac{d^2 \vec{r}}{ds^2} \right\|$. 
Total Curvature Energy $E_c = \int \kappa^2 ds$.
Curvature Frequency is computed via the Fourier transform of the tangent angles along the branch.
**Parameter**: `curvature_smoothing_sigma` (default 2.0). Raw SWC data is noisy; taking derivatives directly amplifies noise. We apply Gaussian smoothing to the coordinates before computing $\kappa$.

### 2.3 Local Inertia Tensor
**Concept**: Describes the spatial distribution of mass (nodes) around a center. Its eigenvalues tell us if a local region is line-like (axon), planar (sheet), or spherical (soma).
**Formula**: 
For a node $p_i$, we define a local neighborhood $N(p_i)$ within `inertia_radius`. The tensor is:
$$ I = \sum_{p_j \in N(p_i)} \left( \|p_j - p_i\|^2 \mathbf{I}_{3x3} - (p_j - p_i)(p_j - p_i)^T \right) $$
We use the sorted eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$ as rotation-invariant features.
**Parameter**: `inertia_radius` (default 10.0). It defines the scale of the "local" neighborhood. Chosen to encompass typical local branching clusters without capturing distant unrelated branches.

### 2.4 Branching Angle and Root Inflow Angle
**Concept**: Angles between incoming and outgoing segments at bifurcation points.
**Formula**: 
For incoming vector $\vec{v}_{in}$ and outgoing vector $\vec{v}_{out}$, the angle $\theta$ is:
$$ \cos(\theta) = \frac{\vec{v}_{in} \cdot \vec{v}_{out}}{\|\vec{v}_{in}\| \|\vec{v}_{out}\|} $$
Angles are naturally translation and rotation invariant.

## 3. Topological Features

### 3.1 Strahler Stream Order
**Concept**: A measure of branching complexity. Leaves have order 1. When two branches of the same order $k$ meet, the parent branch has order $k+1$. If they are different, the parent takes the maximum order.
**Use Case**: Identifies major trunks (high order) vs. terminal dendrites (low order).

### 3.2 Tree Isomorphism (Weisfeiler-Lehman Hashes)
**Concept**: A hash that perfectly identifies the topological structure of a subtree, ignoring node coordinates. It allows the model to recognize repeating biological motifs.

## 4. Modality Masking (Handling Missing Data)
When fragments lack specific modalities (e.g., missing background intensity, or purely topological fragments missing coordinates), we replace the missing vector with a learned token:
- If `background_intensity` is unavailable, its feature slot is filled with `mask_modality_value` (default 0.0), and a binary attention mask signals the model to ignore it or treat it as `[MISSING]`.

## 5. Summary of Tunable Parameters (`config.json`)

Adjusting the parameters in `config.json` allows you to tailor the tokenization strictly to your dataset's resolution and noise levels. Here is the concise logic for tuning them:

- **`resampling_distance`**: 
  - *Logic*: Dictates the spatial resolution. Set this equal to the average internode distance in your cleanest SWC samples.
- **`tortuosity_window_size`**: 
  - *Logic*: Balances micro-wiggles vs overall curve. Increase if your data is highly jittery (digitization noise); decrease if you want to capture very fine biological spine curves.
- **`curvature_smoothing_sigma`**: 
  - *Logic*: Suppresses high-frequency noise before calculating 2nd derivatives. Increase for heavily pixelated skeletons; decrease for ultra-high-resolution tracings.
- **`inertia_radius`**: 
  - *Logic*: The local "receptive field". Set this to approximately the radius of a typical neuronal bifurcation in your dataset to correctly classify branching points from linear axons.
- **`max_tortuosity_allowed`**:
  - *Logic*: Quality control threshold. Set this to slightly above the maximum tortuosity found in your ground-truth data to reject artificially tangled fragments caused by tracing errors.
