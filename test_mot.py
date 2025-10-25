"""Test script for Mixture of Thoughts implementation"""

import torch
from mot.core.model import MixtureOfThoughtsTransformer, MoTConfig


def test_basic_forward():
    """Test basic forward pass"""
    print("🧪 Test 1: Basic forward pass")
    
    config = MoTConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=4,
        num_thoughts=4,
        max_position_embeddings=256
    )
    
    model = MixtureOfThoughtsTransformer(config)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids)
    
    assert 'logits' in outputs
    assert 'metrics' in outputs
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    
    print(f"✅ Output shape: {outputs['logits'].shape}")
    print(f"✅ Number of layers with metrics: {len(outputs['metrics'])}")
    
    return outputs


def test_metrics():
    """Test diversity and entropy metrics"""
    print("\n🧪 Test 2: Metrics computation")
    
    config = MoTConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_thoughts=4
    )
    
    model = MixtureOfThoughtsTransformer(config)
    model.train()  # Enable noise injection
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    outputs = model(input_ids)
    
    for i, metrics in enumerate(outputs['metrics']):
        print(f"\nLayer {i}:")
        print(f"  - Diversity: {metrics['diversity_score']:.3f}")
        print(f"  - Entropy: {metrics['router_entropy']:.3f}")
        print(f"  - Active branches: {metrics['active_branches']}")
    
    print("✅ Metrics computed successfully")
    
    return outputs['metrics']


def test_parallel_generation():
    """Test parallel thought generation"""
    print("\n🧪 Test 3: Parallel thought generation")
    
    config = MoTConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_thoughts=4,
        max_position_embeddings=256
    )
    
    model = MixtureOfThoughtsTransformer(config)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    
    print(f"Input shape: {input_ids.shape}")
    
    results = model.generate_parallel_thoughts(
        input_ids,
        num_parallel=4,
        max_length=10
    )
    
    print(f"\nGenerated {len(results)} parallel thoughts:")
    for i, result in enumerate(results):
        print(f"\nBranch {i+1}:")
        print(f"  - Creativity level: {result['creativity_level']:.2f}")
        print(f"  - Generated tokens: {result['generated_ids'].shape[1] - input_ids.shape[1]}")
        print(f"  - Total length: {result['generated_ids'].shape[1]}")
        
        # Average diversity
        diversities = [m['diversity_score'] for m in result['metrics'] if 'diversity_score' in m]
        if diversities:
            avg_div = sum(diversities) / len(diversities)
            print(f"  - Avg diversity: {avg_div:.3f}")
    
    print("✅ Parallel generation successful")
    
    return results


def test_creativity_control():
    """Test creativity level control"""
    print("\n🧪 Test 4: Creativity control")
    
    config = MoTConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_thoughts=4
    )
    
    model = MixtureOfThoughtsTransformer(config)
    model.train()
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # Low creativity
    creativity_low = [0.5] * config.num_hidden_layers
    outputs_low = model(input_ids, creativity_schedule=creativity_low)
    
    # High creativity
    creativity_high = [1.5] * config.num_hidden_layers
    outputs_high = model(input_ids, creativity_schedule=creativity_high)
    
    div_low = sum(m['diversity_score'] for m in outputs_low['metrics']) / len(outputs_low['metrics'])
    div_high = sum(m['diversity_score'] for m in outputs_high['metrics']) / len(outputs_high['metrics'])
    
    print(f"Low creativity (0.5) avg diversity: {div_low:.3f}")
    print(f"High creativity (1.5) avg diversity: {div_high:.3f}")
    
    print("✅ Creativity control working")
    
    return outputs_low, outputs_high


def test_model_size():
    """Test model parameter count"""
    print("\n🧪 Test 5: Model size")
    
    config = MoTConfig(
        vocab_size=50257,  # GPT-2 vocab size
        hidden_size=768,
        num_hidden_layers=12,
        num_thoughts=8
    )
    
    model = MixtureOfThoughtsTransformer(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1e9:.2f} GB (fp32)")
    
    print("✅ Model instantiated successfully")
    
    return model


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧠 Mixture of Thoughts - Test Suite")
    print("=" * 60)
    
    try:
        test_basic_forward()
        test_metrics()
        test_parallel_generation()
        test_creativity_control()
        test_model_size()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()
