# NeuroGramLM Domain Finetuning

This module orchestrates adapting the pre-trained CCFv3 NeuroGramLM architecture to an entirely new biological domain (e.g., a different animal species or tracing protocol) without destroying the foundational geometric relationships learned during pre-training.

## Finetuning vs Retraining
- **Incremental Retraining (`Neuro_Retraining`)**: New data is strictly within the CCFv3 space. The entire model is updated.
- **Domain Finetuning (`Neuro_Finetuning`)**: New data represents a novel domain. We freeze the core Transformer encoders and inject a `DomainAdaptationKernel` to safely adapt to the new representations.

## How it works
1. **Biological Filter**: Even in a new domain, the physical scaling and extreme bounds must represent plausible biological topologies. The SWCs are normalized and passed through the `BiologicalPlausibilityFilter`.
2. **Encoder Freezing**: The heavy geometric, topological, and biological encoders are completely frozen.
3. **Kernel Injection**: The `FinetuningWrapper` dynamically injects a high-dimensional nonlinear kernel over the `encoder_memory`.
4. **Decoder Adaptation**: Only the injected kernel and the autoregressive decoders are trained, allowing the model to learn the grammar of the new domain without catastrophic forgetting.

## Usage
To finetune on a new domain of SWCs, you must provide the base CCFv3 checkpoint you wish to adapt:

```bash
cd Neuro_Finetuning
python scripts/run_finetuning.py --checkpoint ../checkpoints/latest_ccfv3.pt --resolution 1.0 1.0 3.0
```
*Note: Ensure `config.json` is updated with the path to your new domain dataset.*
