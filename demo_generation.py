"""Demo script for text generation with Mixture of Thoughts"""

import torch
from mot.core.model import MixtureOfThoughtsTransformer, MoTConfig


def demo_parallel_generation():
    """Demo of parallel thought generation"""
    
    print("=" * 70)
    print("🧠 Mixture of Thoughts - Generation Demo")
    print("=" * 70)
    
    # Create small model for demo
    config = MoTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_thoughts=8,
        max_position_embeddings=128
    )
    
    print(f"\nModel configuration:")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Number of layers: {config.num_hidden_layers}")
    print(f"  - Thought branches: {config.num_thoughts}")
    
    # Create model
    model = MixtureOfThoughtsTransformer(config)
    model.eval()
    
    # Create input prompt (random tokens for demo)
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    print(f"\nInput prompt shape: {prompt.shape}")
    print(f"Prompt tokens: {prompt[0].tolist()}")
    
    # Generate with different creativity levels
    print("\n" + "=" * 70)
    print("Generating parallel thoughts with varying creativity...")
    print("=" * 70)
    
    results = model.generate_parallel_thoughts(
        prompt,
        num_parallel=4,
        max_length=20
    )
    
    # Display results
    for i, result in enumerate(results):
        print(f"\n{'=' * 70}")
        print(f"Branch {i+1}:")
        print(f"{'=' * 70}")
        print(f"  Creativity level: {result['creativity_level']:.2f}")
        print(f"  Tokens generated: {result['generated_ids'].shape[1] - prompt.shape[1]}")
        print(f"  Generated sequence: {result['generated_ids'][0].tolist()}")
        
        # Calculate metrics
        diversities = [m.get('diversity_score', 0) for m in result['metrics']]
        entropies = [m.get('router_entropy', 0) for m in result['metrics']]
        
        if diversities:
            avg_diversity = sum(diversities) / len(diversities)
            print(f"  Avg diversity: {avg_diversity:.3f}")
        
        if entropies:
            avg_entropy = sum(entropies) / len(entropies)
            print(f"  Avg router entropy: {avg_entropy:.3f}")
        
        # Branch usage
        branch_counts = {}
        for m in result['metrics']:
            for branch in m.get('active_branches', []):
                branch_counts[branch] = branch_counts.get(branch, 0) + 1
        
        if branch_counts:
            print(f"  Active branches distribution: {dict(sorted(branch_counts.items()))}")
    
    # Compare diversity between branches
    print("\n" + "=" * 70)
    print("Inter-branch diversity analysis:")
    print("=" * 70)
    
    sequences = [result['generated_ids'] for result in results]
    
    # Compare each pair
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            # Simple diversity measure: proportion of different tokens
            diff_count = (sequences[i] != sequences[j]).sum().item()
            total_tokens = sequences[i].shape[1]
            diversity = diff_count / total_tokens
            
            print(f"Branch {i+1} vs Branch {j+1}: {diversity:.1%} different tokens")
    
    print("\n" + "=" * 70)
    print("✅ Generation demo complete!")
    print("=" * 70)
    
    return results


def demo_creativity_control():
    """Demo of creativity control"""
    
    print("\n\n" + "=" * 70)
    print("🎨 Creativity Control Demo")
    print("=" * 70)
    
    config = MoTConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_thoughts=4
    )
    
    model = MixtureOfThoughtsTransformer(config)
    model.train()  # Enable noise injection
    
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    
    # Test different creativity levels
    creativity_levels = [0.3, 0.7, 1.2, 2.0]
    
    print("\nTesting different creativity levels:")
    print("-" * 70)
    
    for creativity in creativity_levels:
        schedule = [creativity] * config.num_hidden_layers
        
        outputs = model(prompt, creativity_schedule=schedule)
        
        # Calculate average diversity
        diversities = [m['diversity_score'] for m in outputs['metrics']]
        avg_div = sum(diversities) / len(diversities)
        
        # Calculate average entropy
        entropies = [m['router_entropy'] for m in outputs['metrics']]
        avg_ent = sum(entropies) / len(entropies)
        
        print(f"Creativity {creativity:.1f}: diversity={avg_div:.3f}, entropy={avg_ent:.3f}")
    
    print("\n✅ Higher creativity → Higher diversity & entropy")


def demo_single_forward():
    """Demo of single forward pass with metrics"""
    
    print("\n\n" + "=" * 70)
    print("📊 Single Forward Pass Demo")
    print("=" * 70)
    
    config = MoTConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=3,
        num_thoughts=6
    )
    
    model = MixtureOfThoughtsTransformer(config)
    model.train()
    
    input_ids = torch.randint(0, config.vocab_size, (2, 15))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    outputs = model(input_ids)
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"\nMetrics per layer:")
    print("-" * 70)
    
    for metrics in outputs['metrics']:
        layer = metrics['layer']
        diversity = metrics['diversity_score']
        entropy = metrics['router_entropy']
        branches = metrics['active_branches']
        
        print(f"Layer {layer}: diversity={diversity:.3f}, entropy={entropy:.3f}, branches={branches}")
    
    print("\n✅ Each layer shows different routing patterns!")


if __name__ == "__main__":
    # Run all demos
    demo_parallel_generation()
    demo_creativity_control()
    demo_single_forward()
    
    print("\n\n" + "=" * 70)
    print("🎉 All demos completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Train on real data (Shakespeare, WikiText, etc.)")
    print("2. Compare with baseline models")
    print("3. Analyze emergent specialization patterns")
    print("4. Fine-tune with LoRA for efficiency")
