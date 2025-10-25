# Mixture of Thoughts (MoT)

**Non-Deterministic Parallel Reasoning for Language Models**

Mixture of Thoughts (MoT) is a novel architecture that enables language models to explore multiple reasoning paths simultaneously through stochastic branching, inspired by both Mixture of Experts (MoE) and diffusion models.

## 🧠 Key Concepts

Unlike traditional autoregressive generation (token-by-token) or deterministic Mixture of Experts, MoT introduces:

- **Probabilistic Thought Routing**: Non-deterministic selection of reasoning branches
- **Creative Noise Injection**: Stochastic exploration of the semantic space
- **Parallel Thought Processing**: Multiple reasoning paths executed simultaneously
- **Intelligent Combination**: Attention-based merging of diverse thoughts

## 🚀 Quick Start

```python
import torch
from mot import MixtureOfThoughtsTransformer, MoTConfig

# Create model
config = MoTConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_thoughts=8
)
model = MixtureOfThoughtsTransformer(config)

# Generate parallel thoughts
input_ids = torch.randint(0, config.vocab_size, (1, 10))
results = model.generate_parallel_thoughts(
    input_ids, 
    num_parallel=4, 
    max_length=50
)

# Analyze diversity
for i, result in enumerate(results):
    print(f"Branch {i+1}: Creativity={result['creativity_level']:.2f}")
```

## � Project Structure

```
mixture-of-thoughts/
├── mot/                    # Core package
│   ├── core/              # Model components (branch, router, layer, model)
│   └── __init__.py
├── configs/               # Configuration files
│   ├── vm/               # VM deployment configs (H100, A100, L4, T4)
│   └── training/         # Training configs (optimized per GPU)
├── scripts/              # All executable scripts
│   ├── deploy_gcp.sh     # GCP VM deployment
│   ├── deploy_with_config.sh  # Config-based deployment
│   ├── train_mot.py      # Main training (supports multi-GPU DDP)
│   ├── train_with_config.py   # Config-based training wrapper
│   ├── train_distributed.sh   # Multi-GPU launcher
│   └── demo_generation.py     # Generation demo
├── docs/                 # Documentation
├── tests/                # Test suite
├── environment.yml       # Conda environment
└── requirements.txt      # Python dependencies
```

## �📦 Installation

### Local Setup
```bash
pip install -r requirements.txt
# or
bash scripts/setup_env.sh
```

### GCP Deployment
```bash
# Deploy with H100 GPU
./scripts/deploy_with_config.sh configs/vm/h100.yaml

# Deploy 4x A100 with large model
./scripts/deploy_with_config.sh configs/vm/a100_multi.yaml configs/training/large_multigpu.yaml

# Monitor training
./scripts/deploy_gcp.sh monitor
```

## 🏗️ Architecture

### Core Components

1. **ThoughtRouter**: Probabilistic routing to select active branches
2. **ThoughtBranch**: Non-deterministic processing with noise injection
3. **MixtureOfThoughtsLayer**: Combines routing, branching, and merging
4. **AttentionCombiner**: Intelligently merges outputs from multiple branches

### Differences from MoE

| Feature | MoE | MoT |
|---------|-----|-----|
| **Goal** | Efficiency | Exploration |
| **Selection** | Top-K deterministic | Probabilistic sampling |
| **Experts** | Specialized FFNs | Stochastic branches |
| **Output** | Single path | Multiple diverse paths |
| **Training** | Minimize loss | Maximize diversity + quality |

## 📊 Metrics

MoT provides real-time metrics:

- **Diversity Score**: 1 - cosine similarity between branches
- **Router Entropy**: Measure of selection randomness
- **Active Branches**: Number of branches used per sample

## 🔬 Research Applications

- **Creative Writing**: Explore multiple narrative directions
- **Reasoning Tasks**: Consider different solution approaches
- **Code Generation**: Generate diverse implementations
- **Scientific Discovery**: Explore hypothesis space

## 🏋️ Training

### Local Training
```bash
# Small model (~30 min on T4)
python scripts/train_with_config.py --config configs/training/small.yaml

# Large model
python scripts/train_with_config.py --config configs/training/h100.yaml
```

### Multi-GPU Training
```bash
# Automatically detects GPUs and uses PyTorch DDP
./scripts/train_distributed.sh configs/training/large_multigpu.yaml
```

### Available Configs

**VM Configs** (`configs/vm/`):
- `t4.yaml` - T4 16GB ($0.35/hr) - Testing
- `l4.yaml` - L4 24GB ($0.75/hr) - Medium models
- `a100.yaml` - A100 40GB ($3.67/hr) - Large models
- `h100.yaml` - H100 80GB ($5/hr) - Extra large models
- `*_multi.yaml` - 4-GPU configurations

**Training Configs** (`configs/training/`):
- `small.yaml` - 30M params (256 hidden, 4 layers)
- `l4.yaml` - 150M params (512 hidden, 8 layers)
- `a100.yaml` - 250M params (768 hidden, 10 layers)
- `h100.yaml` - 400M params (1024 hidden, 12 layers)
- `large_multigpu.yaml` - 900M params (1536 hidden, 16 layers)

## 📚 Examples

See `notebooks/` for detailed examples:

- `01_quick_start.ipynb`: Basic usage and concepts
- `02_lora_finetuning.ipynb`: Adapting existing models with LoRA
- `03_analysis.ipynb`: Analyzing thought patterns and diversity

Run demo:
```bash
python scripts/demo_generation.py
```

## 🔧 Adapting Existing Models

MoT can be grafted onto existing transformers via LoRA (coming soon):

```python
from mot.adapters import adapt_model_with_lora
from transformers import GPT2LMHeadModel

base_model = GPT2LMHeadModel.from_pretrained("gpt2")
mot_model = adapt_model_with_lora(base_model, num_thoughts=8)
```

## 📖 Citation

```bibtex
@article{mixtureofthoughts2025,
  title={Mixture of Thoughts: Non-Deterministic Parallel Reasoning via Stochastic Expert Branching},
  author={},
  year={2025}
}
```

## 🤝 Contributing

Contributions welcome! See issues for current tasks.

## 📄 License

MIT License
