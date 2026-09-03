# NeuroGramLM Continuous Retraining

This module orchestrates the incremental fine-tuning of the model when new SWC datasets are acquired. It is designed to handle both complete whole-brain neurons as well as individual disconnected **neuron fragments**. 

## The Challenge
NeuroGramLM's vector quantization (VQ) vocabulary was trained strictly on the topological distributions and geometric scales of neurons within the **Allen CCFv3 space**. If a new batch of neurons is traced at an arbitrary scale, or is heavily distorted, training on them directly would destroy the latent representations of the model (catastrophic forgetting).

## The Biological Plausibility Filter
To ensure that the model maps new data relative to the existing CCFv3 space without "forgetting", the `run_incremental.py` script employs a strict **Biological Plausibility Gating Mechanism**.

Before any training occurs:
1. **Scale Harmonization:** The script automatically scales the bounding box of the new SWCs to match CCFv3 standards.
2. **Metric Validation:** The script mocks tokenizing the SWCs to measure intrinsic relative properties (like tortuosity and curvature energy).
3. **The Gate:** If these intrinsic metrics explode beyond the bounds set in `config.json` (meaning the geometry of the neuron is physically implausible in CCFv3 space), **the script rejects the SWC**. It will *not* be added to the training set.

This guarantees that the model only learns from data that shares the same relative biological meaning as the original training set.

## Usage
To process a folder of raw SWCs and incrementally update the latest model checkpoint:

```bash
cd Neuro_Retraining
python scripts/run_incremental.py
```
*Note: Make sure `config.json` has `io_paths.raw_swc_input_dir` pointing to your new data folder.*
