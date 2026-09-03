"""
Author: samik1986
Date: 2026-09-03
"""
import argparse
import json
import os
import subprocess
import sys
from utils.logger import get_logger

logger = get_logger("NeuroGram_CLI", module_name="Global")

def load_global_config():
    config_path = os.path.join(os.path.dirname(__file__), 'global_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def update_module_config(module_dir, updates):
    """
    Overwrites specific keys in a sub-module's config.json based on global CLI args.
    """
    config_path = os.path.join(os.path.dirname(__file__), module_dir, 'config.json')
    if not os.path.exists(config_path):
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Naive dict update logic for flat overrides
    for k, v in updates.items():
        if v is not None:
            # We would need proper mapping depending on the module's config structure
            pass
            
    # For now, we pass CLI args directly to the subprocess or rely on the master config.
    # In a full production setup, this func would map global args -> local config.json keys.

def run_subprocess(command_list, cwd=None):
    logger.info(f"Executing: {' '.join(command_list)}")
    if cwd is None:
        cwd = os.path.dirname(__file__)
        
    try:
        subprocess.run(command_list, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline Step Failed with exit code {e.returncode}.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="NeuroGramLM Unified Pipeline Orchestrator")
    parser.add_argument('--step', type=str, required=True, 
                        choices=['tokenize', 'train', 'infer', 'retrain', 'finetune'],
                        help="The pipeline step to execute.")
                        
    # Global Overrides
    parser.add_argument('--input_dir', type=str, default=None, help="Input directory (raw SWCs).")
    parser.add_argument('--epochs', type=int, default=None, help="Number of epochs for training/retraining.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint to resume/infer from.")
    parser.add_argument('--source_swc', type=str, default=None, help="Source SWC for inference bridging.")
    parser.add_argument('--tiff_volume', type=str, default=None, help="Bio volume TIFF for inference bridging.")
    parser.add_argument('--resolution', type=float, nargs=3, default=None, help="XYZ physical resolution (microns/voxel)")
    
    args = parser.parse_args()
    
    global_config = load_global_config()
    logger.info(f"Initialized NeuroGramLM Orchestrator. Requested Step: {args.step.upper()}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.step == 'tokenize':
        logger.info("Routing to Tokenization Pipeline...")
        script_path = os.path.join(base_dir, 'Neuro_Tokenization', 'run_pipeline.py')
        run_subprocess([sys.executable, script_path])
        
    elif args.step == 'train':
        logger.info("Routing to Training Pipeline...")
        script_path = os.path.join(base_dir, 'Neuro_Training', 'scripts', 'train.py')
        cmd = [sys.executable, script_path]
        if args.checkpoint:
            cmd.extend(["--resume_from", args.checkpoint])
        run_subprocess(cmd)
        
    elif args.step == 'infer':
        logger.info("Routing to Inference Engine...")
        script_path = os.path.join(base_dir, 'Neuro_Training', 'scripts', 'inference.py')
        cmd = [sys.executable, script_path]
        if args.source_swc:
            cmd.extend(['--source_swc', args.source_swc])
        if args.tiff_volume:
            cmd.extend(['--tiff_volume', args.tiff_volume])
        if args.resolution:
            cmd.extend(['--resolution', str(args.resolution[0]), str(args.resolution[1]), str(args.resolution[2])])
            
        run_subprocess(cmd)
        
    elif args.step == 'retrain':
        logger.info("Routing to Incremental Retraining Pipeline...")
        script_path = os.path.join(base_dir, 'Neuro_Retraining', 'scripts', 'run_incremental.py')
        cmd = [sys.executable, script_path]
        if args.resolution:
            cmd.extend(['--resolution', str(args.resolution[0]), str(args.resolution[1]), str(args.resolution[2])])
            
        run_subprocess(cmd)
        
    elif args.step == 'finetune':
        logger.info("Routing to Domain Finetuning Pipeline...")
        script_path = os.path.join(base_dir, 'Neuro_Finetuning', 'scripts', 'run_finetuning.py')
        cmd = [sys.executable, script_path]
        if args.checkpoint:
            cmd.extend(['--checkpoint', args.checkpoint])
        else:
            logger.error("Finetuning requires a --checkpoint to finetune from.")
            sys.exit(1)
            
        if args.resolution:
            cmd.extend(['--resolution', str(args.resolution[0]), str(args.resolution[1]), str(args.resolution[2])])
            
        run_subprocess(cmd)
        
    logger.info("Orchestrator finished successfully.")

if __name__ == "__main__":
    main()
