# How to Update NeuroGramLM Architecture

This codebase was designed specifically with modularity in mind, ensuring that as the biology and tokenization requirements evolve, the model can scale without requiring a total rewrite.

## 1. Bio Tower Updates (Training Mode)
Currently, the Bio Tower (incorporating a 3D CNN) operates strictly in **zero-shot inference mode** because the training dataset lacks image intensity information. 
If labeled training data with intensity patches becomes available in the future:
1. **Model Modification:** Go to `model.py` -> `forward()`. Remove the `if 'bio_volumes' in batch:` fallback logic that zeroes out the Bio Encoding during training.
2. **Loss Update:** Ensure that the decoder loss functions (or fusion losses) correctly calculate gradients backward through `bio_enc`.

## 2. Changing Decoder Autoregressive Behavior
The Multi-Tower Decoder is defined in `layers/decoders.py`. 
- If you wish to decode Geometry and Topology *dependently* (e.g., predicting the shape based on the topology of the current token), you would pass the output of the `TopologicalDecoder` as an additional memory mask or cross-attention input into the `GeometricDecoder`. Currently, they decode in parallel, independent multi-tower streams.

## 3. Changing the Loss Functions
Because losses are separated in the `losses/` directory:
- To change how the model evaluates autoregressive generation (e.g., switching from Cross-Entropy to a specialized hierarchical tree loss for topology), simply edit `losses/decoder_loss.py`. The `model.py` orchestrator does not need to be changed as long as the inputs/outputs match.

## 4. Changing Layer Stacking & CNN Parameters
- To adjust the CNN patch sizes or channels, simply modify the `cnn_3d` block in `config.json`.
- To add Cross-Attention instead of Concatenation in the fusion block, edit the `__init__` and `forward` of `MultimodalFusion` in `layers/fusion.py`.
- To increase the depth of the towers or decoders, adjust `n_layers_per_encoder` and `n_layers_per_decoder` in `config.json`.
