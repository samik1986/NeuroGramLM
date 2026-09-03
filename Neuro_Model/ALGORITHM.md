# NeuroGramLM Architectural Algorithm

## The Data Flow (Encoder-Decoder)

1. **Input Generation:** 
   - **Training:** The model receives Geometric and Topological sequences of existing fragments. Biological volumetric patches (`bio_volumes`) are masked/omitted because the training SWC graph dataset lacks imaging intensity info.
   - **Inference/Test:** For gap detection and bridging, the model receives the Fragment 1 sequence, Fragment 2 sequence, and raw 3D intensity patches extracted along the ridge/gap between them.

2. **Embedding Phase (`layers/embeddings.py`):**
   - **Geometric:** VQ IDs -> Embeddings -> Sum = `G_emb`
   - **Topological:** Discrete integers -> Embeddings -> Sum = `T_emb`
   - **Biological (Zero-Shot):** 3D CNN processes the `(D, H, W)` image patches into dense feature vectors -> Linear projection = `B_emb`.

3. **Multi-Tower Encoding & Fusion (`layers/towers.py`, `layers/fusion.py`):**
   - Each `*_emb` passes through its respective independent Encoder Tower, utilizing Rotary Position Embedding (RoPE) to track spatial sequence.
   - The Geometric and Topological encodings are fused into a shared `encoder_memory` latent representation.
   - At inference, the Bio encoding is integrated into the memory pool, allowing the model to dynamically "see" the intensity ridge.

4. **Multi-Tower Decoding (`layers/decoders.py`):**
   - To bridge a gap, the decoder must predict the missing sequence of tokens autoregressively.
   - The decoder uses two independent towers: `GeometricDecoder` and `TopologicalDecoder`.
   - **Cross-Attention:** The decoders cross-attend to the `encoder_memory` (which at test time includes the zero-shot Bio intensity features).
   - This setup forces the decoder to output both the physical shape and tree hierarchy required to link the fragments.

5. **Decoder Losses (`losses/decoder_loss.py`):**
   - Trained using Teacher Forcing on continuous neuron segments by artificially splitting them and asking the decoder to fill the gap.
   - **Modality Dropout:** By training the decoder without the Bio features, it learns robust shape imputation purely from geometry/topology. This guarantees that at test time, the injection of zero-shot Bio ridge data only acts to *validate* and *refine* the generated bridge, preventing over-reliance on missing modalities.
