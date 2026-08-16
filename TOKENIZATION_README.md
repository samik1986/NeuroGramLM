# Biological Branching Grammar: Tokenization Pipeline

This repository contains the algorithm and code required to convert massive datasets of raw, unaligned 3D neuron graphs (SWC files) into linear, translation-independent sequences for Transformer Language Model training (NeuroGramLM).

## The Algorithm

The algorithm converts 3D geometric graphs into a "SMILES-like" flat sequence using **Depth-First Search (DFS)** and **Vector Quantization (VQ)**.

### Phase A: Vocabulary Training
Instead of using explicit $(X, Y, Z)$ coordinates (which fail on unregistered tissues and floating sub-neurites), we use relative branch geometry.

1. **Vector Extraction**: For every physical branch edge $u \to v$, we compute the relative geometric vector $\Delta \vec{v} = (\Delta X, \Delta Y, \Delta Z, \Delta R)$.
2. **Stratified Sampling**: To train the vocabulary, we take a stratified random sample of SWC files across different brain region directories.
3. **K-Means Vector Quantization**: The extracted $\Delta \vec{v}$ vectors are subsampled and passed through K-Means clustering. Each cluster center forms a geometric vocabulary token: `<GEO_0>` to `<GEO_N>`.

*The trained vocabulary model is saved to disk so it does not need to be recomputed.*

### Phase B: Iterative Sequence Generation & Stitching
With the vocabulary trained, we iterate through the full 180,000 neuron dataset using multi-core multiprocessing.

1. **Fast Vectorized Prediction**: For a given SWC file, all physical edges are translated into their nearest `<GEO_X>` cluster ID simultaneously using a vectorized NumPy/scikit-learn predict step.
2. **Iterative DFS**: A Depth-First Search traverses the graph. 
   - When a branch splits, it emits a `<BIF>` token.
   - When traversing down a branch, it emits `<GEO_X>` tokens.
   - When it hits a dead-end leaf node, it emits `<POP>` to backtrack.
   - Cycle protection (`visited` sets) strictly enforces Eulerian paths in dirty graphs.
3. **Spatial Jump Tokens (Union / Intersection)**: If the SWC contains disconnected subtrees (e.g. broken neurites across stitched imaging tiles), a `<JUMP_x_y_z>` token is emitted between them. 
   - The jump is the quantized coordinate distance from the *last visited leaf node* of Subtree A to the *root node* of Subtree B.
   - This allows the attention mechanism to seamlessly stitch floating neurons.

## Configuration Parameters (`config.json`)

| Parameter | Description |
| :--- | :--- |
| `SWC_DIR` | The root directory containing your `.swc` dataset files. |
| `NUM_CLUSTERS` | The size of the geometric vocabulary (e.g., `128` or `512`). |
| `JUMP_BIN_SIZE` | The physical resolution grid for jump tokens (e.g., `10.0` micrometers). Balances vocabulary size vs stitching precision. |
| `VOCAB_TRAIN_FILES` | The number of SWC files to randomly sample for training the vocabulary model (e.g., `1000`). |
| `MAX_SAMPLES_FOR_KMEANS` | The hard limit on vectors fed into K-Means to ensure fast training times (e.g., `100000`). |
| `MAX_WORKERS` | Number of CPU cores to use for Phase B tokenization. |
| `OUTPUT_FILE` | The resulting sequence dataset. Streamed to `.jsonl` to ensure $O(1)$ memory usage. |

## Running the Pipeline

Ensure you have your environment set up and the required dependencies (`scikit-learn`, `joblib`, `numpy`) installed.

```bash
python tokenize_swc_vq.py
```
