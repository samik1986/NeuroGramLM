# Neuro Dataloader Module

This directory contains the data loading pipeline for the NeuroGramLM architecture. It handles the ingestion of tokenized neuronal data from JSON files, sequence chunking, data imputation, and tensor formatting for autoregressive training.

## Functions

### `NeuroDataset(Dataset)`
- **File**: `dataset.py`
- **Description**: Parses `.json` token streams from the tokenization output. Given a list of files, it iteratively opens them and converts the geometric `vq_ids` and topological `embeddings` into PyTorch tensors. It returns a dictionary containing `inputs` and `targets` correctly shifted by one time step for causal modeling.
- **Sequence Chunking**: To prevent CUDA Out-of-Memory (OOM) errors from extremely long fragments, the dataset randomly crops token sequences to a maximum length (`max_seq_len`).
- **Imputation**: Any null values found in the input embeddings (e.g. `background_intensity`) are gracefully defaulted to a safe value.

### `neuro_collate_fn(batch)`
- **File**: `collate.py`
- **Description**: Batches multiple sequences of varying lengths. It applies zero-padding up to the longest sequence in the batch.
- **Mask Generation**: Autogenerates the boolean `padding_mask` for the Encoder blocks and the triangular causal `tgt_mask` for the Decoder blocks.

## Assumptions
- The raw JSON files strictly follow the token structure: `tokens -> vq_ids` & `tokens -> embeddings`.
- Time steps are sequential inside the JSON array.
- The vocab size for the `wl_hash` embedding matches the model config, requiring a modulo operation during parsing.

## Configuration Parameters
All dataset and collator logic is driven by the local `config.json` inside this folder.

- **`dataset_parameters`**:
  - `max_seq_len` (int): The maximum number of tokens allowed per sample sequence before it gets randomly chunked. (Default: 2048)
  - `null_imputation_value` (int): The fallback index to use when an expected embedding is explicitly `null` in the data stream. (Default: 0)
  - `wl_vocab_size` (int): The vocabulary cap to modulo the topological Weisfeiler-Lehman hash against. (Default: 1000)
- **`collate_parameters`**:
  - `padding_idx_inputs` (int): The value used to pad sequences for the encoder input. This is ignored by the transformer's `padding_mask`. (Default: 0)
  - `padding_idx_targets` (int): The value used to pad target sequences. Must align with the `ignore_index` of your PyTorch `CrossEntropyLoss`. (Default: -1)

## How to Update

Do **not** edit the hardcoded padding or lengths directly in the Python scripts. 
To adjust memory footprint, batching rules, or imputation behavior, open `Neuro_Dataloader/config.json` and change the corresponding values. Both `dataset.py` and `collate.py` read from this configuration dynamically at runtime. If you add new data types to the training pipeline, add their respective settings to `config.json` first before parsing them.
