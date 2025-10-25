#!/bin/bash
# Setup script for Mixture of Thoughts environment with Miniconda

set -e  # Exit on error

echo "🧠 Setting up Mixture of Thoughts environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="mot"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Exiting..."
        exit 0
    fi
fi

# Create new environment with Python 3.10
echo "📦 Creating conda environment '${ENV_NAME}' with Python 3.10..."
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
echo "✅ Environment created. Activating..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install PyTorch (adjust based on your system)
echo "🔥 Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon
        echo "   Detected Apple Silicon (M1/M2/M3)"
        conda install pytorch torchvision torchaudio -c pytorch -y
    else
        # Intel Mac
        echo "   Detected Intel Mac"
        conda install pytorch torchvision torchaudio -c pytorch -y
    fi
else
    # Linux - check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "   Detected NVIDIA GPU, installing CUDA version"
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    else
        echo "   No GPU detected, installing CPU version"
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
fi

# Install other dependencies from requirements.txt
echo "📚 Installing other dependencies..."
# Fix PyTorch dependencies
pip install "sympy==1.13.1" fsspec
pip install transformers>=4.30.0
pip install numpy>=1.24.0
pip install tqdm>=4.65.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install jupyter>=1.0.0
pip install datasets>=2.12.0
pip install wandb>=0.15.0
pip install peft>=0.5.0

# Install development tools
echo "🛠️  Installing development tools..."
pip install pytest black isort flake8

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "To test the installation, run:"
echo "   python test_mot.py"
echo ""
echo "To deactivate the environment, run:"
echo "   conda deactivate"
