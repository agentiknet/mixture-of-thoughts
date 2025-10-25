#!/usr/bin/env python3
"""Fast training script for MoT - nanoGPT-inspired approach v2

Key fixes from nanoGPT analysis:
1. Pin memory and non_blocking GPU transfers
2. Async data prefetching during forward pass
3. Proper autocast context usage
4. Fetch first batch before loop starts
5. Don't eval on iter 0 in main loop
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from pathlib import Path
from contextlib import nullcontext

from mot.core.model import MixtureOfThoughtsTransformer, MoTConfig


def load_data(data_path: str):
    """Load and prepare data using numpy array"""
    print(f"Loading data from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    data = np.array([char_to_idx[ch] for ch in text], dtype=np.uint16)
    
    print(f"Loaded {len(data):,} tokens, {len(chars)} unique chars")
    
    return data, len(chars), char_to_idx


def get_batch(data, batch_size, seq_length, device, device_type):
    """Get a batch with pin_memory and non_blocking transfer - nanoGPT style"""
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    
    x = torch.stack([torch.from_numpy(data[i:i+seq_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+seq_length].astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # Pin memory for async GPU transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y


def compute_loss(outputs, labels):
    """Simple cross-entropy loss"""
    logits = outputs['logits']
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    return loss


@torch.no_grad()
def estimate_loss(model, data, eval_iters, batch_size, seq_length, device, device_type, ctx):
    """Estimate loss over multiple batches"""
    model.eval()
    losses = []
    
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, seq_length, device, device_type)
        with ctx:
            outputs = model(x)
            loss = compute_loss(outputs, y)
        losses.append(loss.item())
    
    model.train()
    return np.mean(losses)


def main(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Autocast context (use bfloat16 if available, else float16)
    if device_type == 'cuda':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        print(f"Using autocast with {dtype}")
    else:
        ctx = nullcontext()
    
    # Load data
    train_data, vocab_size, char_to_idx = load_data(args.train_file)
    
    # Model config
    config = MoTConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_thoughts=args.num_thoughts,
        max_position_embeddings=args.seq_length
    )
    
    # Create model
    model = MixtureOfThoughtsTransformer(config).to(device)
    
    # Compile if requested (nanoGPT does this)
    if args.use_compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # GradScaler for fp16
    use_amp = (device_type == 'cuda' and dtype == 'float16')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Training setup
    iter_num = 0
    best_val_loss = float('inf')
    
    num_iters = args.max_iters if args.max_iters > 0 else (
        (len(train_data) // (args.batch_size * args.seq_length)) * args.num_epochs
    )
    
    print(f"\nTraining for {num_iters:,} iterations")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Eval interval: {args.eval_interval}")
    print("=" * 70)
    
    # CRITICAL: Fetch first batch before loop starts (nanoGPT style)
    X, Y = get_batch(train_data, args.batch_size, args.seq_length, device, device_type)
    
    model.train()
    t0 = time.time()
    local_iter_num = 0
    
    while iter_num < num_iters:
        # Evaluate periodically (but not on iter 0 in main loop)
        if iter_num % args.eval_interval == 0 and iter_num > 0:
            val_loss = estimate_loss(
                model, train_data, args.eval_iters, 
                args.batch_size, args.seq_length, device, device_type, ctx
            )
            
            print(f"step {iter_num}: val loss {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    checkpoint_path = output_dir / "best_model.pt"
                    torch.save({
                        'iter': iter_num,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'config': config.__dict__
                    }, checkpoint_path)
        
        # Forward pass with autocast
        with ctx:
            outputs = model(X)
            loss = compute_loss(outputs, Y)
        
        # CRITICAL: Async prefetch next batch DURING forward pass (nanoGPT style)
        X, Y = get_batch(train_data, args.batch_size, args.seq_length, device, device_type)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if args.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % args.log_interval == 0:
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        
        iter_num += 1
        local_iter_num += 1
    
    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    
    # Save final model
    if args.output_dir:
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "final_model.pt"
        torch.save({
            'iter': iter_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__
        }, checkpoint_path)
        print(f"Saved final model to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast MoT training (nanoGPT-style v2)")
    
    # Data
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--seq_length", type=int, default=128)
    
    # Model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_thoughts", type=int, default=8)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Optimization
    parser.add_argument("--use_compile", action="store_true")
    
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=1)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/fast_run_v2")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)
