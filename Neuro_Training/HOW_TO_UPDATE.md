# How to Update NeuroGramLM Training & Inference

## 1. Changing Hyperparameters
All critical parameters are controlled via `config.json`.
- **Training Setup:** Adjust `batch_size`, `learning_rate`, `epochs`, and gradient clipping in the `training_parameters` block.
- **Inference Setup:** To make the bridging search faster (but potentially less accurate), decrease `knn_top_k`. To make it stricter, increase `bio_validation_threshold`.

## 2. Updating TIFF/SWC Inference Preprocessing
Inference runs on raw data. The `InferencePreprocessor` in `inference.py` handles this.
- **TIFF Extraction:** Currently, `tifffile` is used (or mocked) to extract 3D intensity ridges. If your volume data transitions to HDF5 or Zarr (for out-of-core massive brain processing), update the `extract_tiff_patch` method to use `h5py` or `zarr` libraries instead of `tifffile`.
- **Scale Normalization:** We currently force raw SWC inputs to have a bounding box diagonal of 500 microns to match CCFv3 space (preventing curvature/inertia metric explosion). If you prefer a different normalization heuristic (e.g., matching median edge length), update `normalize_scale()` in `inference.py`.

## 3. Modifying the Loss Weights
If the model is struggling to learn the topological tree structure but excels at the geometric shape, you can dynamically scale the loss backpropagation.
In `config.json`, navigate to `training_parameters.loss_weights` and increase `topological` relative to `geometric` or `decoder`.

## 4. Updating the Training Loop
- `scripts/train.py` contains the standard PyTorch training loop.
- If you need to add specialized learning rate schedulers (like Cosine Annealing with Warmup) or mixed-precision training (`torch.cuda.amp`), instantiate them inside `main()` and wrap the forward/backward passes inside `train_epoch()`.

## 5. Generating Loss Graphs
The `analysis/loss_graph.py` script automatically parses standard JSONL logs.
If you add a new loss metric to `train.py` (e.g., a dedicated biological intensity contrastive loss once labels are acquired), you must:
1. Update `train_epoch()` to log it in the output dictionary.
2. Update `parse_logs()` in `loss_graph.py` to extract the new key.
3. Add a new `plt.plot()` line in `plot_loss_curves()`.
