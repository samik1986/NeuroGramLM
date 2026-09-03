# Training & Inference Algorithm Workflow

## 1. The Training Loop (`train.py`)
1. **Data Prep:** The model receives input batches of pre-tokenized JSONs (Geometry + Topology). Biological data is explicitly set to `None`.
2. **Forward Pass:** The data passes through the respective encoder towers, gets fused into `encoder_memory`, and is passed to the decoders.
3. **Autoregression:** The `GeometricDecoder` and `TopologicalDecoder` predict the shifted targets using Cross-Attention over the `encoder_memory`.
4. **Loss Calculation:** We aggregate:
   - Auxiliary Geometric Loss (predicting shape from the Geom Tower)
   - Auxiliary Topological Loss (predicting hierarchy from the Topo Tower)
   - Fusion Loss
   - Main Decoder Loss (autoregressive generation)
5. **Backpropagation:** Gradients are clipped to 1.0 to prevent explosion, and `AdamW` steps the weights.

## 2. The Bridging Inference (`inference.py`)
Because the dataset is massive, the model must bridge disconnected neurons efficiently. **Inference operates on raw biology files rather than pre-tokenized dicts.**

### Pre-Processing (Raw Data Ingestion)
1. **Scale Normalization:** Raw SWC files are ingested alongside an optional `(x, y, z)` physical resolution. The preprocessor multiplies the SWC points by this resolution. To prevent metrics from exploding out of VQ codebook bounds, the SWC is then bounded so its diagonal matches the CCFv3 training baseline (e.g., ~500 microns).
2. **Tokenization:** Geometric features (tortuosity, curvature energy) are computed. Because these metrics measure relative bending, the data is completely translation and **rotation invariant**.
3. **Encoding:** The fragment is mapped to the token vocabulary.

### Stage 1: Decoder Prediction & Fast Search
1. The source fragment (Fragment A) is passed through the Encoder -> Decoder.
2. The Decoder predicts the latent vector for the *first node of the missing gap*.
3. We perform a Fast KNN Search against a pre-indexed vector database containing the starting node representations of all disconnected SWC fragments in the dataset.
4. We retrieve the Top-10 closest matches.

### Stage 2: Zero-Shot Biological Validation
1. For each of the Top-10 candidates (Fragment B), we compute the physical line between the end of A and the start of B.
2. **TIFF Extraction:** We open the raw `.tif` Bio Volume file and physically extract a 3D cropped tensor (e.g., `32x32x32`) around this trajectory.
3. This `bio_volume` tensor is passed into the model alongside Fragment A's geometry/topology.
4. The model's `BioTower` (3D CNN + RoPE) encodes the TIFF intensity ridge and injects it into the `encoder_memory`.
5. A final confidence score is generated. The candidate with the highest validation score > Threshold is chosen as the true continuation of the neuron.
