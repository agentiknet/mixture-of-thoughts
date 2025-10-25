"""Training script for Mixture of Thoughts model"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import wandb

from mot.core.model import MixtureOfThoughtsTransformer, MoTConfig


class TextDataset(Dataset):
    """Simple text dataset for training"""
    
    def __init__(self, file_path: str, seq_length: int = 128, vocab_size: int = 50257):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Load text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple character-level tokenization for demo
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Encode text
        self.data = [self.char_to_idx[ch] for ch in text]
        
        print(f"Dataset loaded: {len(self.data)} tokens, {len(self.chars)} unique characters")
    
    def __len__(self):
        return len(self.data) - self.seq_length - 1
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels


def compute_loss(outputs, labels, diversity_weight=0.1, entropy_weight=0.05):
    """Compute combined loss with diversity and entropy terms"""
    
    # Language modeling loss
    logits = outputs['logits']
    lm_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Diversity loss (average across layers)
    diversity_scores = [m['diversity_score'] for m in outputs['metrics']]
    diversity_loss = -sum(diversity_scores) / len(diversity_scores)
    
    # Entropy loss (average across layers)
    entropy_scores = [m['router_entropy'] for m in outputs['metrics']]
    entropy_loss = -sum(entropy_scores) / len(entropy_scores)
    
    # Combined loss
    total_loss = lm_loss + diversity_weight * diversity_loss + entropy_weight * entropy_loss
    
    return {
        'total_loss': total_loss,
        'lm_loss': lm_loss.item(),
        'diversity_loss': diversity_loss,
        'entropy_loss': entropy_loss
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, diversity_weight, entropy_weight):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_diversity = 0
    total_entropy = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(input_ids)
        
        # Compute loss
        loss_dict = compute_loss(outputs, labels, diversity_weight, entropy_weight)
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss_dict['total_loss'].item()
        total_lm_loss += loss_dict['lm_loss']
        total_diversity += loss_dict['diversity_loss']
        total_entropy += loss_dict['entropy_loss']
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total_loss'].item():.4f}",
            'lm': f"{loss_dict['lm_loss']:.4f}",
            'div': f"{loss_dict['diversity_loss']:.4f}"
        })
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'train/loss': loss_dict['total_loss'].item(),
                'train/lm_loss': loss_dict['lm_loss'],
                'train/diversity_loss': loss_dict['diversity_loss'],
                'train/entropy_loss': loss_dict['entropy_loss'],
                'train/learning_rate': scheduler.get_last_lr()[0]
            })
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'lm_loss': total_lm_loss / n_batches,
        'diversity': total_diversity / n_batches,
        'entropy': total_entropy / n_batches
    }


def evaluate(model, dataloader, device, diversity_weight, entropy_weight):
    """Evaluate model"""
    
    model.eval()
    total_loss = 0
    total_lm_loss = 0
    
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids)
            loss_dict = compute_loss(outputs, labels, diversity_weight, entropy_weight)
            
            total_loss += loss_dict['total_loss'].item()
            total_lm_loss += loss_dict['lm_loss']
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'lm_loss': total_lm_loss / n_batches
    }


def main(args):
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="mixture-of-thoughts",
            name=args.run_name,
            config=vars(args)
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create config
    config = MoTConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_thoughts=args.num_thoughts,
        max_position_embeddings=args.seq_length
    )
    
    # Create model
    model = MixtureOfThoughtsTransformer(config)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset = TextDataset(args.train_file, seq_length=args.seq_length, vocab_size=args.vocab_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    if args.eval_file:
        eval_dataset = TextDataset(args.eval_file, seq_length=args.seq_length, vocab_size=args.vocab_size)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        eval_loader = None
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            args.diversity_weight, args.entropy_weight
        )
        
        print(f"\nTraining metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  LM Loss: {train_metrics['lm_loss']:.4f}")
        print(f"  Diversity: {train_metrics['diversity']:.4f}")
        print(f"  Entropy: {train_metrics['entropy']:.4f}")
        
        # Evaluate
        if eval_loader is not None:
            eval_metrics = evaluate(model, eval_loader, device, args.diversity_weight, args.entropy_weight)
            print(f"\nEvaluation metrics:")
            print(f"  Loss: {eval_metrics['loss']:.4f}")
            print(f"  LM Loss: {eval_metrics['lm_loss']:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    'eval/loss': eval_metrics['loss'],
                    'eval/lm_loss': eval_metrics['lm_loss'],
                    'epoch': epoch
                })
            
            # Save best model
            if eval_metrics['loss'] < best_loss:
                best_loss = eval_metrics['loss']
                checkpoint_path = output_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': config.__dict__
                }, checkpoint_path)
                print(f"  ✅ Saved best model to {checkpoint_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    print(f"\n{'='*70}")
    print("✅ Training complete!")
    print(f"{'='*70}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mixture of Thoughts model")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval_file", type=str, default=None, help="Path to evaluation data")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    
    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_thoughts", type=int, default=8, help="Number of thought branches")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--diversity_weight", type=float, default=0.1, help="Diversity loss weight")
    parser.add_argument("--entropy_weight", type=float, default=0.05, help="Entropy loss weight")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--run_name", type=str, default="mot_training", help="Run name for wandb")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    main(args)
