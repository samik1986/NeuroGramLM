files = {
    'skeletonize_volume.py': 'Unsupervised skeleton fragments in extracted_fragments.swc',
    'inference_routing.py': 'Traced optimal path connecting fragments',
    'transformer_model.py': 'Multi-stream Transformer Neural Network model',
    'train.py': 'Trained NeuroGram Transformer model weights',
    'dataset.py': 'PyTorch DataLoader with Graph Laplacian Positional Encoding'
}
inputs = {
    'skeletonize_volume.py': 'Raw TIFF volume (e.g., config.json specified)',
    'inference_routing.py': 'Vesselness map and two coordinates (p1, p2)',
    'transformer_model.py': 'Tokenized NeuroGram data',
    'train.py': 'SWC tokenized dataset',
    'dataset.py': 'JSONL tokenized dataset'
}

for fname, output_desc in files.items():
    with open(fname, 'r') as f:
        content = f.read()
    if 'Author: Samik Banerjee' in content:
        continue
    
    header = f'\"\"\"\nAuthor: Samik Banerjee\nDate of creation: August 2026\nHow to run it: python {fname}\nInput: {inputs[fname]}\nOutput: {output_desc}\n\"\"\"\n\n'
    
    with open(fname, 'w') as f:
        f.write(header + content)
print('Headers added.')
