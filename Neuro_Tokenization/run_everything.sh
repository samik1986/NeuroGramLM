#!/bin/bash
set -e
echo "====================================="
echo "Step 1: Building VQ Codebooks"
echo "====================================="
python3 build_vq_vocab.py

echo "====================================="
echo "Step 2: Running Full Pipeline"
echo "====================================="
python3 run_pipeline.py
