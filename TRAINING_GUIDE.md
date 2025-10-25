# Mixture of Thoughts - Training Guide

Complete guide for training MoT models locally or on cloud infrastructure.

---

## Quick Start (Local)

### 1. Download Sample Data

```bash
# Create data directory
mkdir -p data

# Tiny Shakespeare (1.1MB) - using curl (macOS/Linux)
curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Or create your own text file
echo "Your training text here..." > data/train.txt
```

### 2. Train Locally

```bash
# Small model for testing (fast)
python train_mot.py \
  --train_file data/shakespeare.txt \
  --vocab_size 100 \
  --hidden_size 256 \
  --num_layers 4 \
  --num_thoughts 8 \
  --batch_size 16 \
  --num_epochs 10 \
  --output_dir outputs/shakespeare_small

# Full-size model (requires GPU)
python train_mot.py \
  --train_file data/shakespeare.txt \
  --vocab_size 50257 \
  --hidden_size 768 \
  --num_layers 12 \
  --num_thoughts 8 \
  --batch_size 32 \
  --num_epochs 20 \
  --use_wandb \
  --output_dir outputs/shakespeare_full
```

### 3. Monitor Training

If using `--use_wandb`:
1. Install: `pip install wandb`
2. Login: `wandb login`
3. View at: https://wandb.ai/

---

## Cloud Training (GCP)

### Option A: Google Compute Engine VM

**Recommended for: Full control, custom setups**

#### 1. Create VM with GPU

```bash
# Set your project
export PROJECT_ID="your-project-id"
export ZONE="us-central1-a"
export VM_NAME="mot-training"

# Create VM with T4 GPU (cheapest)
gcloud compute instances create $VM_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

**Cost estimate**: ~$0.35/hour (T4) to ~$2.50/hour (V100)

#### 2. SSH and Setup

```bash
# SSH into VM
gcloud compute ssh $VM_NAME --zone=$ZONE

# Clone your code
git clone https://github.com/yourusername/mixture-of-thoughts.git
cd mixture-of-thoughts

# Setup environment
source setup_env.sh
conda activate mot

# Install additional dependencies
pip install wandb tqdm
```

#### 3. Upload Data

```bash
# From local machine
gcloud compute scp data/shakespeare.txt $VM_NAME:~/mixture-of-thoughts/data/ --zone=$ZONE

# Or download directly on VM (curl works on most systems)
curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

#### 4. Start Training

```bash
# Run in tmux (survives disconnect)
tmux new -s training

# Start training
python train_mot.py \
  --train_file data/shakespeare.txt \
  --vocab_size 50257 \
  --hidden_size 768 \
  --num_layers 12 \
  --batch_size 32 \
  --num_epochs 50 \
  --use_wandb \
  --output_dir outputs/gcp_run

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

#### 5. Download Results

```bash
# From local machine
gcloud compute scp --recurse $VM_NAME:~/mixture-of-thoughts/outputs/ ./outputs/ --zone=$ZONE
```

#### 6. Stop VM (Important!)

```bash
# Stop to avoid charges
gcloud compute instances stop $VM_NAME --zone=$ZONE

# Delete when done
gcloud compute instances delete $VM_NAME --zone=$ZONE
```

---

### Option B: Google Colab (Free/Pro)

**Recommended for: Quick experiments, no setup**

#### 1. Create Notebook

```python
# Install dependencies
!pip install torch transformers peft wandb tqdm

# Clone repo
!git clone https://github.com/yourusername/mixture-of-thoughts.git
%cd mixture-of-thoughts

# Download data
!curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train
!python train_mot.py \
  --train_file data/shakespeare.txt \
  --vocab_size 1000 \
  --hidden_size 512 \
  --num_layers 6 \
  --batch_size 16 \
  --num_epochs 20 \
  --output_dir outputs/colab_run
```

**Limitations**:
- Free: T4 GPU, 12h timeout
- Pro: V100/A100, 24h timeout
- May disconnect randomly

---

### Option C: Vertex AI Training

**Recommended for: Production, managed training**

#### 1. Package Code

```bash
# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="mixture-of-thoughts",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
    ],
)
EOF

# Build package
python setup.py sdist
```

#### 2. Upload to GCS

```bash
export BUCKET_NAME="your-bucket-name"
export REGION="us-central1"

# Create bucket
gcloud storage buckets create gs://$BUCKET_NAME --location=$REGION

# Upload data
gcloud storage cp data/shakespeare.txt gs://$BUCKET_NAME/data/

# Upload code
gcloud storage cp dist/mixture-of-thoughts-0.1.0.tar.gz gs://$BUCKET_NAME/code/
```

#### 3. Submit Training Job

```bash
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=mot-training-1 \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/deeplearning-platform-release/pytorch-gpu.1-13 \
  --args="--train_file=gs://$BUCKET_NAME/data/shakespeare.txt,--output_dir=gs://$BUCKET_NAME/outputs/"
```

**Cost**: Similar to Compute Engine, but more managed

---

## Cloud Cost Comparison

| Platform | GPU | Cost/Hour | Best For |
|----------|-----|-----------|----------|
| GCP Compute (T4) | T4 | $0.35 | Experimentation |
| GCP Compute (V100) | V100 | $2.50 | Fast training |
| GCP Compute (A100) | A100 | $3.67 | Large models |
| Colab Free | T4 | Free | Quick tests |
| Colab Pro | V100/A100 | $10/mo | Regular use |
| Vertex AI | Various | Variable | Production |

**Example training costs**:
- Small model (4 layers, 1h): ~$0.35
- Medium model (12 layers, 4h): ~$1.40
- Large model (24 layers, 12h): ~$4.20

---

## Training Tips

### 1. Start Small

```bash
# Test with tiny model first (5 minutes)
python train_mot.py \
  --train_file data/shakespeare.txt \
  --vocab_size 100 \
  --hidden_size 128 \
  --num_layers 2 \
  --num_thoughts 4 \
  --batch_size 8 \
  --num_epochs 5
```

### 2. Monitor GPU Usage

```bash
# On VM/local with GPU
watch -n 1 nvidia-smi

# Expected GPU utilization: 70-90%
# If lower: increase batch_size or num_workers
```

### 3. Adjust Hyperparameters

```bash
# High diversity (creative writing)
--diversity_weight 0.2 \
--entropy_weight 0.1

# Low diversity (factual tasks)
--diversity_weight 0.05 \
--entropy_weight 0.02

# More thought branches (slower but more diverse)
--num_thoughts 16
```

### 4. Resume from Checkpoint

```python
# In train_mot.py, add loading logic:
if args.resume_from:
    checkpoint = torch.load(args.resume_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

---

## Next Steps

1. **Download real datasets**:
   - WikiText: `curl -O https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip`
   - OpenWebText: https://huggingface.co/datasets/openwebtext
   
2. **Fine-tune with LoRA** (coming soon): `train_lora_mot.py`

3. **Evaluate diversity**: `evaluate_diversity.py` (coming soon)

4. **Deploy API**: `serve_mot.py` (coming soon)

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch_size 8

# Or use gradient accumulation
--gradient_accumulation_steps 4

# Or smaller model
--hidden_size 512 --num_layers 6
```

### Slow Training

```bash
# Use mixed precision (2x faster)
--fp16

# More data workers
--num_workers 8

# Larger batch size (if memory allows)
--batch_size 64
```

### NaN Loss

```bash
# Lower learning rate
--learning_rate 5e-5

# Gradient clipping (already enabled at 1.0)
# Increase warmup
--warmup_ratio 0.2
```

---

## Support

- Issues: https://github.com/yourusername/mixture-of-thoughts/issues
- Wandb: https://wandb.ai/
- GCP Docs: https://cloud.google.com/compute/docs
