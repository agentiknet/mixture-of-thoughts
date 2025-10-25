#!/bin/bash
# Quick setup script to run on GCP VM
# Usage: curl -sSL https://raw.githubusercontent.com/agentiknet/mixture-of-thoughts/main/vm_setup.sh | bash

set -e

echo "========================================="
echo "🚀 MoT Training - VM Setup"
echo "========================================="

# Install system dependencies
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl tmux build-essential

# Install miniconda
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

# Initialize conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Configure conda - use conda-forge to avoid TOS issues
echo "Configuring conda..."
conda config --add channels conda-forge
conda config --set channel_priority strict
conda config --remove channels defaults 2>/dev/null || true

# Clone repo
echo "Cloning repository..."
cd $HOME
if [ ! -d "mixture-of-thoughts" ]; then
    git clone https://github.com/agentiknet/mixture-of-thoughts.git
else
    echo "Repository exists, pulling latest..."
    cd mixture-of-thoughts
    git pull
    cd $HOME
fi

cd $HOME/mixture-of-thoughts

# Create environment
echo "Creating conda environment..."
if ! conda env list | grep -q "^mot "; then
    conda create -n mot python=3.10 -y
fi

# Activate environment
conda activate mot

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA..."
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
echo "Installing dependencies..."
pip install --quiet transformers peft wandb tqdm sympy==1.13.1 fsspec

# Download data
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
echo "  conda activate mot"
echo "  ./start_training.sh"
echo ""
echo "Or manually:"
echo "  python train_mot.py \\"
echo "    --train_file data/shakespeare.txt \\"
echo "    --vocab_size 50257 \\"
echo "    --hidden_size 512 \\"
echo "    --num_layers 6 \\"
echo "    --batch_size 8 \\"
echo "    --num_epochs 20 \\"
echo "    --output_dir outputs/run1"
