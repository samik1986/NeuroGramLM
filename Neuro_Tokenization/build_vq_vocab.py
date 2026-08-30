import os
import json
import random
import logging
from core.pipeline import TokenizationPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def get_all_swc_files(directory):
    swc_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.swc'):
                swc_files.append(os.path.join(root, file))
    return swc_files

def main():
    logging.info("Initializing VQ Codebook Builder...")
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    pipeline = TokenizationPipeline(config)
    
    # 1. Stratified Random Sampling
    swc_dir = config['io_paths']['input_swc_dir']
    sample_ratio = config['vector_quantization']['sample_ratio']
    
    logging.info(f"Scanning {swc_dir} for SWC files...")
    all_files = get_all_swc_files(swc_dir)
    
    sample_size = max(1, int(len(all_files) * sample_ratio))
    sampled_files = random.sample(all_files, sample_size)
    
    logging.info(f"Sampled {len(sampled_files)} files ({sample_ratio*100}%) for training codebooks.")
    
    from run_pipeline import extract_fragments_from_swc
    
    # 2. Extract features and fit MiniBatchKMeans incrementally
    processed_fragments = 0
    for i, swc_file in enumerate(sampled_files):
        fragments = extract_fragments_from_swc(swc_file)
        for frag in fragments:
            if len(frag) < config['quality_thresholds']['min_nodes_per_fragment']:
                continue
                
            features = pipeline.extract_features(frag)
            pipeline.vq.fit_partial(features)
            processed_fragments += 1
            
        if (i + 1) % 100 == 0:
            logging.info(f"Fitted {i+1}/{len(sampled_files)} files (approx {processed_fragments} fragments)...")
            
    # 3. Save Codebooks
    pipeline.vq.flush_buffer()
    pipeline.vq.is_trained = True
    pipeline.vq.save()
    logging.info(f"Successfully trained and saved Multimodal VQ Codebooks to {config['vector_quantization']['model_dir']}")

if __name__ == '__main__':
    main()
