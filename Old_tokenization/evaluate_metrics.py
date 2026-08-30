"""
Author: Samik Banerjee
Date of creation: August 2026
How to run it: python evaluate_metrics.py
Input: tokenized_dataset.jsonl and original .swc files
Output: Random sample topological preservation metrics printed to console
"""
import os
import glob
import random
import subprocess
import json
import numpy as np
import sys

def run_evaluation():
    print("Evaluating tokenization metrics...")
    if not os.path.exists("tokenized_dataset.jsonl"):
        print("tokenized_dataset.jsonl not found!")
        return

    sample_size = 100
    print(f"Sampling {sample_size} neurons for evaluation using reservoir sampling...")
    
    sampled_lines = []
    with open("tokenized_dataset.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i < sample_size:
                sampled_lines.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    sampled_lines[j] = line
    
    with open("tokenized_dataset_eval.jsonl", "w") as f:
        for line in sampled_lines:
            f.write(line)
            
    with open("config.json", "r") as f:
        config = json.load(f)
        
    orig_output_file = config.get("OUTPUT_FILE", "tokenized_dataset.jsonl")
    config["OUTPUT_FILE"] = "tokenized_dataset_eval.jsonl"
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("Running detokenization on sample...")
    subprocess.run([sys.executable, "detokenize_swc.py"], check=True)
    
    config["OUTPUT_FILE"] = orig_output_file
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    detokenized_files = glob.glob("detokenized_*.swc")
    print(f"Generated {len(detokenized_files)} detokenized files. Comparing...")
    
    sys.path.append('.')
    import compare_swc
    
    total_len_diff_pct = []
    node_matches = 0
    
    for detok_file in detokenized_files:
        orig_filename = detok_file.replace("detokenized_", "")
        orig_file = compare_swc.find_original_swc(orig_filename, search_dir="SWCs/")
        
        if not orig_file:
            continue
            
        orig_stats = compare_swc.parse_swc_stats(orig_file)
        detok_stats = compare_swc.parse_swc_stats(detok_file)
        
        if orig_stats and detok_stats:
            if orig_stats['num_nodes'] == detok_stats['num_nodes']:
                node_matches += 1
                
            diff_len = abs(orig_stats['total_length'] - detok_stats['total_length'])
            pct_len = (diff_len / orig_stats['total_length']) * 100 if orig_stats['total_length'] > 0 else 0
            total_len_diff_pct.append(pct_len)
            
    for f in detokenized_files:
        os.remove(f)
    os.remove("tokenized_dataset_eval.jsonl")
        
    if total_len_diff_pct:
        print("\n" + "="*50)
        print("FINAL EVALUATION METRICS (Based on random sample)")
        print("="*50)
        print(f"Topological Preservation (Exact Node Matches): {node_matches}/{len(total_len_diff_pct)} ({(node_matches/len(total_len_diff_pct))*100:.1f}%)")
        print(f"Average Total Path Length Shrinkage: {np.mean(total_len_diff_pct):.2f}%")
        print("="*50)
        print("Evaluation Complete!")
    else:
        print("Failed to run comparison metrics.")

if __name__ == "__main__":
    run_evaluation()
