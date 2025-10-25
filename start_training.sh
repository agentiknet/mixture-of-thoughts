#!/bin/bash
# Start training in tmux session
# Optimized for V100 16GB GPU

set -e

# Use absolute path for conda
CONDA_PATH="$HOME/miniconda3/bin/conda"

# Check if conda exists
if [ ! -f "$CONDA_PATH" ]; then
    echo "Error: Conda not found at $CONDA_PATH"
    exit 1
fi

# Check if conda environment exists
if ! $CONDA_PATH env list | grep -q "^mot "; then
    echo "Error: mot environment not found. Run vm_setup.sh first!"
    exit 1
fi

# Start tmux session with training
tmux new-session -d -s training bash -c '
    # Source conda initialization
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate mot
    cd ~/mixture-of-thoughts
    
    echo "Starting training with V100-optimized settings..."
    echo "Model: 512 hidden, 6 layers (~100M params)"
    echo "Batch: 8, Epochs: 20"
    echo ""
    
    # Verify conda environment is active
    echo "Python: $(which python)"
    echo "PyTorch installed: $(python -c \"import torch; print(torch.__version__)\" 2>&1)"
    echo ""
    
    python train_mot.py \
        --train_file data/shakespeare.txt \
        --vocab_size 50257 \
        --hidden_size 512 \
        --num_layers 6 \
        --num_thoughts 8 \
        --batch_size 8 \
        --num_epochs 20 \
        --learning_rate 1e-4 \
        --diversity_weight 0.1 \
        --entropy_weight 0.05 \
        --output_dir outputs/v100_run \
        2>&1 | tee training.log
    
    # Keep session alive on error
    echo ""
    echo "Training finished or exited. Press Enter to close."
    read
'

echo "✅ Training started in tmux session 'training'"
echo ""
echo "To monitor:"
echo "  tmux attach -t training"
echo ""
echo "To detach without stopping: Ctrl+B, then D"
echo ""
echo "To check status:"
echo "  tail -f ~/mixture-of-thoughts/training.log"
