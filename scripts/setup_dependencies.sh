#!/bin/bash
# Setup dependencies on GCP VM
# Run this inside the mot conda environment

set -e

echo "========================================="
echo "🚀 Installing MoT Dependencies"
echo "========================================="
echo ""

# Check if in mot environment
if [[ "$CONDA_DEFAULT_ENV" != "mot" ]]; then
    echo "Error: Please activate mot environment first:"
    echo "  conda activate mot"
    exit 1
fi

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies..."
pip install transformers peft wandb tqdm sympy==1.13.1 fsspec pyyaml

# Download training data
echo "Downloading training data..."
mkdir -p data
if [ ! -f "data/shakespeare.txt" ]; then
    curl -sL -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi

echo ""
echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo ""
echo "To start training:"
echo "  python scripts/train_with_config.py --config configs/training/small.yaml"
