#!/bin/bash
# Run training on GCP VM with specified config
# Usage: ./run_training_on_vm.sh <training_config>

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TRAINING_CONFIG="${1:-configs/training/small.yaml}"

if [ ! -f "$TRAINING_CONFIG" ]; then
    echo "Error: Config file not found: $TRAINING_CONFIG"
    exit 1
fi

echo "========================================"
echo "🚀 Starting Training on VM"
echo "========================================"
echo "Config: $TRAINING_CONFIG"
echo ""

# Source conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Error: Conda not found. Run vm_setup.sh first."
    exit 1
fi

# Activate environment
if ! conda env list | grep -q "^mot "; then
    echo "Creating conda environment..."
    conda config --add channels conda-forge 2>/dev/null || true
    conda config --set channel_priority strict 2>/dev/null || true
    conda config --remove channels defaults 2>/dev/null || true
    conda create -n mot python=3.10 -y
    
    conda activate mot
    
    # Install dependencies
    echo "Installing PyTorch..."
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    echo "Installing other dependencies..."
    pip install --quiet transformers peft wandb tqdm sympy==1.13.1 fsspec pyyaml
else
    conda activate mot
fi

# Download data if needed
mkdir -p data
if [ ! -f "data/shakespeare.txt" ]; then
    echo "Downloading training data..."
    curl -sL -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi

# Determine if multi-GPU training is needed
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")

echo ""
echo "Configuration:"
echo "  GPUs detected: $GPU_COUNT"
echo "  Training config: $TRAINING_CONFIG"
echo ""

# Start training in tmux session
echo "Starting training in tmux session 'training'..."

if [ "$GPU_COUNT" -gt 1 ]; then
    # Multi-GPU training with torchrun
    TRAIN_CMD="./scripts/train_distributed.sh $TRAINING_CONFIG"
else
    # Single GPU training
    TRAIN_CMD="python scripts/train_with_config.py --config $TRAINING_CONFIG"
fi

# Kill existing training session if any
tmux kill-session -t training 2>/dev/null || true

# Start new training session
tmux new-session -d -s training "
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate mot
    cd ~/mixture-of-thoughts
    $TRAIN_CMD 2>&1 | tee training.log
"

echo ""
echo -e "${GREEN}✓ Training started!${NC}"
echo ""
echo "To monitor training:"
echo "  tmux attach -t training"
echo ""
echo "To detach from tmux without stopping:"
echo "  Press Ctrl+B, then D"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "To view logs:"
echo "  tail -f training.log"
