#!/usr/bin/env python3
"""Inference script for MoT model - generate text from trained checkpoint"""

import torch
import numpy as np
import argparse
from pathlib import Path

from mot.core.model import MixtureOfThoughtsTransformer, MoTConfig


def load_checkpoint(checkpoint_path: str, device):
    """Load trained checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    config = MoTConfig(**checkpoint['config'])
    
    # Create model and load weights
    model = MixtureOfThoughtsTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from iter {checkpoint.get('iter', 'unknown')}")
    print(f"Val loss: {checkpoint.get('val_loss', 'unknown')}")
    
    return model, config


def load_vocab(data_path: str):
    """Load vocabulary from training data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    
    return char_to_idx, idx_to_char


@torch.no_grad()
def generate(model, prompt, char_to_idx, idx_to_char, max_new_tokens, temperature=1.0, top_k=None, device='cuda'):
    """Generate text from prompt"""
    # Encode prompt
    prompt_ids = [char_to_idx.get(ch, 0) for ch in prompt]
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    generated = prompt_ids.copy()
    
    for _ in range(max_new_tokens):
        # Crop to max sequence length if needed
        x_cond = x if x.size(1) <= model.config.max_position_embeddings else x[:, -model.config.max_position_embeddings:]
        
        # Forward pass
        outputs = model(x_cond)
        logits = outputs['logits']
        
        # Get last token logits and apply temperature
        logits = logits[:, -1, :] / temperature
        
        # Optional top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        x = torch.cat([x, next_token], dim=1)
        generated.append(next_token.item())
    
    # Decode
    return ''.join([idx_to_char.get(idx, '?') for idx in generated])


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Load vocabulary
    char_to_idx, idx_to_char = load_vocab(args.vocab_file)
    print(f"Vocabulary size: {len(char_to_idx)}\n")
    
    # Generate
    print("=" * 70)
    print("GENERATING TEXT")
    print("=" * 70)
    print(f"Prompt: '{args.prompt}'")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("=" * 70)
    print()
    
    generated_text = generate(
        model, 
        args.prompt, 
        char_to_idx, 
        idx_to_char, 
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print(generated_text)
    print()
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with trained MoT model")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to training data (for vocab)")
    parser.add_argument("--prompt", type=str, default="ROMEO:", help="Text prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (e.g. 40)")
    
    args = parser.parse_args()
    main(args)
