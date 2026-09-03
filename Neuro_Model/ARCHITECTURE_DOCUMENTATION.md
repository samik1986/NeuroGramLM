# NeuroGramLM System Architecture

This document provides a comprehensive structural view of the NeuroGramLM architecture, detailing the inputs/outputs, the flow of data through the individual PyTorch components, and the applied loss functions.

## 1. High-Level Architecture Diagram (Encoder-Decoder)

The architecture uses a full multi-tower Encoder-Decoder mechanism. During training, the Bio Tower is bypassed due to missing data. At test time, the Decoder relies on the Bio Tower's zero-shot intensity ridge encodings to validate generated gaps.

```mermaid
graph TD
    %% Inputs
    subgraph Inputs
        A1[Geometric Tokens]
        A2[Topological Tokens]
        A3[Bio 3D Intensity Patches]
    end

    %% Embedding Layer
    subgraph "Embeddings (layers/embeddings.py)"
        B1[GeometricEmbedding]
        B2[TopologicalEmbedding]
        B3[BioEmbedding<br>3D CNN]
    end

    %% Encoder Towers
    subgraph "Encoder Towers (layers/towers.py)"
        C1[GeometricTower + RoPE]
        C2[TopologicalTower + RoPE]
        C3[BioTower + RoPE<br>Zero-Shot inference]
    end

    %% Fusion
    subgraph "Multimodal Fusion (layers/fusion.py)"
        D1[Concatenate Features]
        D2[Linear Projection]
        D3[Encoder Self-Attention]
    end

    %% Memory
    E1((Fused Encoder Memory))

    %% Decoders
    subgraph "Decoder Towers (layers/decoders.py)"
        F1[GeometricDecoder]
        F2[TopologicalDecoder]
    end

    %% Auxiliary Heads
    subgraph "Autoregressive Projection Heads"
        H1[Geometric Next-Token Predictor]
        H2[Topological Next-Token Predictor]
    end

    %% Losses
    subgraph "Decoder Losses (losses/decoder_loss.py)"
        L1((Cross-Entropy Loss))
    end

    %% Flow
    A1 --> B1
    A2 --> B2
    A3 -.-> B3
    
    B1 --> C1
    B2 --> C2
    B3 -.-> C3
    
    C1 --> D1
    C2 --> D1
    
    D1 --> D2
    D2 --> D3
    D3 --> E1
    C3 -.-> E1
    
    E1 --> F1
    E1 --> F2
    
    F1 --> H1
    F2 --> H2
    
    H1 --> L1
    H2 --> L1
    
    style A3 stroke-dasharray: 5 5, fill:#eee, stroke:#999
    style B3 stroke-dasharray: 5 5, fill:#eee, stroke:#999
    style C3 stroke-dasharray: 5 5, fill:#eee, stroke:#999
```

---

## 2. Component Descriptions & Flow

### A. Inputs & Outputs (IO)
**Inputs (`batch`):**
- **Geometric `vq_ids`**: Integer tensors for tortuosity, curvature, inertia.
- **Topological IDs**: Integer tensors for Strahler, WL Hash.
- **`bio_volumes`**: A `(batch, seq_len, 1, D, H, W)` volumetric tensor representing the raw image intensity patches along a gap/ridge.

**Outputs:**
- `geom_logits`: Predictions for the next geometric VQ IDs in the sequence.
- `topo_logits`: Predictions for the next topological integers.

### B. Embeddings (`layers/embeddings.py`)
- **Use:** Maps discrete integer inputs and continuous volume patches into dense `d_model` vectors.
- **Mechanism:** Geometric/Topological tokens use standard embedding lookups. The `BioEmbedding` utilizes a **3D CNN** to extract spatial features from the volumetric image patches before projecting them to `d_model`.

### C. Multi-Tower Encoders & Fusion (`layers/towers.py`, `layers/fusion.py`)
- **Use:** Encodes the fragment context.
- **Mechanism:** Geometric and Topological streams process independently and are then fused into `encoder_memory`. At test time (zero-shot), the `BioTower` encodes the intensity ridge between fragments and adds it to the memory pool.

### D. Multi-Tower Decoders (`layers/decoders.py`)
- **Use:** Generates the bridging sequence connecting two fragments.
- **Mechanism:** Uses masked self-attention over previously generated tokens, and **cross-attention** over the `encoder_memory` (which includes the Zero-Shot Bio features at inference time). It predicts both geometry and topology concurrently using independent decoder towers.

### E. Decoder Loss (`losses/decoder_loss.py`)
- **Use:** Autoregressive training.
- **Mechanism:** Standard Cross-Entropy applied over the decoder logits against shifted target sequences.
