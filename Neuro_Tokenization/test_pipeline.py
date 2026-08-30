"""
Author: Samik Banerjee
Date: 2026-08-30
Usage: Run this script to test the execution of the modular tokenization pipeline.
"""

import json
import numpy as np
from core.pipeline import TokenizationPipeline

def test_pipeline():
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    pipeline = TokenizationPipeline(config)
    
    # Create a dummy noisy straight line (mock neuron fragment)
    points = np.linspace(0, 10, 10)
    # Add random noise for tortuosity
    points = np.array([[x, x*0.1 + np.random.normal(0, 0.1), np.random.normal(0, 0.1)] for x in points])
    
    # Run the pipeline
    print("--- Running Pipeline ---")
    result = pipeline.process_fragment(points)
    
    if result['success']:
        print(f"\nSuccess! Generated {len(result['tokens'])} tokens.")
        print(f"Quality Score: {result['quality_score']}")
        print(f"Modality Completeness: {result['metrics']['modality_completeness']}")
        
        # Test appending a modality
        print("\n--- Testing Appending Modality ---")
        dummy_background = [0.8 for _ in range(10)]
        updated_tokens = pipeline.append_modality(result['tokens'], 'background_intensity', dummy_background)
        print(f"Token 0 keys after appending: {updated_tokens[0]['embeddings'].keys()}")
    else:
        print("Pipeline Failed.")

if __name__ == "__main__":
    test_pipeline()
