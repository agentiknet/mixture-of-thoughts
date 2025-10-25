#!/bin/bash
# Multi-GPU distributed training launcher for Mixture of Thoughts

set -e

# Get configuration file
CONFIG_FILE="${1:-configs/training/large_multigpu.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 <config_file>"
    exit 1
fi

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi

echo "========================================================================="
echo "🚀 Launching Distributed Training"
echo "========================================================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo ""

# Launch distributed training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_with_config.py \
    --config "$CONFIG_FILE"

echo ""
echo "✅ Training complete!"
