#!/usr/bin/env python3
"""Train MoT model from YAML config file"""

import yaml
import argparse
import sys
from pathlib import Path

# Import the existing train_mot module
from train_mot import main as train_main


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config: dict):
    """Convert config dict to argparse Namespace"""
    
    class Args:
        def __init__(self, config):
            # Required args
            self.train_file = config['train_file']
            self.eval_file = config.get('eval_file', None)
            self.seq_length = config.get('seq_length', 128)
            
            # Model args
            self.vocab_size = config['vocab_size']
            self.hidden_size = config['hidden_size']
            self.num_layers = config['num_layers']
            self.num_thoughts = config['num_thoughts']
            
            # Training args
            self.batch_size = config['batch_size']
            self.num_epochs = config['num_epochs']
            self.learning_rate = config['learning_rate']
            self.weight_decay = config.get('weight_decay', 0.01)
            self.warmup_ratio = config.get('warmup_ratio', 0.1)
            self.diversity_weight = config['diversity_weight']
            self.entropy_weight = config['entropy_weight']
            
            # Output args
            self.output_dir = config['output_dir']
            self.save_every = config.get('save_every', 5)
            self.run_name = config.get('run_name', 'mot_training')
            self.use_wandb = config.get('use_wandb', False)
            
            # System args
            self.num_workers = config.get('num_workers', 4)
            self.seed = config.get('seed', 42)
    
    return Args(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MoT from config file")
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file (e.g., configs/small.yaml)"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    
    cmd_args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(cmd_args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Override settings if specified
    if cmd_args.use_wandb:
        config['use_wandb'] = True
    if cmd_args.batch_size:
        config['batch_size'] = cmd_args.batch_size
    
    print("\nConfiguration:")
    print("-" * 50)
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print("-" * 50)
    print()
    
    # Convert to args and run training
    args = config_to_args(config)
    train_main(args)
