#!/bin/bash
# Automated GCP VM deployment and training for Mixture of Thoughts

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "🚀 Mixture of Thoughts - GCP Training Deployment"
echo "========================================================================"

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-mot-training-$(date +%s)}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"

# Training configuration
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_LAYERS="${NUM_LAYERS:-12}"

# Check for required tools
command -v gcloud >/dev/null 2>&1 || { echo -e "${RED}Error: gcloud CLI not installed${NC}"; exit 1; }

# Get project ID if not set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}No GCP_PROJECT_ID set. Using current gcloud config...${NC}"
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}Error: No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID${NC}"
        exit 1
    fi
fi

echo ""
echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Zone: $ZONE"
echo "  VM Name: $VM_NAME"
echo "  Machine Type: $MACHINE_TYPE"
echo "  GPU: $GPU_TYPE x$GPU_COUNT"
echo ""

# Function to create VM
create_vm() {
    echo -e "${GREEN}Creating VM with GPU...${NC}"
    
    gcloud compute instances create $VM_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-standard \
        --maintenance-policy=TERMINATE \
        --metadata="install-nvidia-driver=True" \
        --scopes=https://www.googleapis.com/auth/cloud-platform
    
    echo -e "${GREEN}✓ VM created successfully${NC}"
    
    # Wait for VM to be ready
    echo "Waiting for VM to be ready (60 seconds)..."
    sleep 60
}

# Function to setup VM
setup_vm() {
    echo -e "${GREEN}Setting up environment on VM...${NC}"
    
    # Create setup script
    cat > /tmp/setup_mot.sh << 'SETUP_SCRIPT'
#!/bin/bash
set -e

echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl tmux build-essential

# Install NVIDIA drivers if not already installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y -qq ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    echo "Waiting for driver installation to complete..."
    sleep 10
fi

# Install miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

# Initialize conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Accept conda ToS
echo "Accepting conda Terms of Service..."
conda config --set allow_conda_downgrades true
$HOME/miniconda3/bin/conda config --set channel_priority flexible

# Clone public repo
echo "Cloning repository..."
cd $HOME
if [ ! -d "mixture-of-thoughts" ]; then
    git clone https://github.com/agentiknet/mixture-of-thoughts.git
else
    echo "Repository already exists, pulling latest..."
    cd mixture-of-thoughts
    git pull
fi
cd $HOME/mixture-of-thoughts

# Create conda environment (using mamba for speed)
echo "Creating conda environment..."
$HOME/miniconda3/bin/conda create -n mot python=3.10 -y -c conda-forge
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
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
curl -sL -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

echo "✓ Setup complete!"
SETUP_SCRIPT

    # Copy setup script to VM
    gcloud compute scp /tmp/setup_mot.sh $VM_NAME:~/setup_mot.sh --zone=$ZONE --project=$PROJECT_ID
    
    # Run setup script
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID --command="bash ~/setup_mot.sh"
    
    echo -e "${GREEN}✓ Environment setup complete${NC}"
}

# Function to upload code
upload_code() {
    echo -e "${GREEN}Uploading code to VM...${NC}"
    
    # Create list of files to upload
    FILES_TO_UPLOAD=(
        "mot/"
        "train_mot.py"
        "requirements.txt"
    )
    
    for file in "${FILES_TO_UPLOAD[@]}"; do
        if [ -e "$file" ]; then
            gcloud compute scp --recurse "$file" $VM_NAME:~/mixture-of-thoughts/ --zone=$ZONE --project=$PROJECT_ID
        fi
    done
    
    echo -e "${GREEN}✓ Code uploaded${NC}"
}

# Function to start training
start_training() {
    echo -e "${GREEN}Starting training...${NC}"
    
    # Create training script
    cat > /tmp/start_training.sh << TRAIN_SCRIPT
#!/bin/bash
set -e

cd ~/mixture-of-thoughts

# Activate conda environment
eval "\$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mot

# Start training in tmux
tmux new-session -d -s training "
    python train_mot.py \\
        --train_file data/shakespeare.txt \\
        --vocab_size 50257 \\
        --hidden_size $HIDDEN_SIZE \\
        --num_layers $NUM_LAYERS \\
        --batch_size $BATCH_SIZE \\
        --num_epochs $TRAIN_EPOCHS \\
        --use_wandb \\
        --output_dir outputs/gcp_run \\
        2>&1 | tee training.log
"

echo "Training started in tmux session 'training'"
echo "To attach: tmux attach -t training"
echo "To detach: Ctrl+B, then D"
TRAIN_SCRIPT

    # Upload and run training script
    gcloud compute scp /tmp/start_training.sh $VM_NAME:~/start_training.sh --zone=$ZONE --project=$PROJECT_ID
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID --command="bash ~/start_training.sh"
    
    echo -e "${GREEN}✓ Training started!${NC}"
    echo ""
    echo "Training is running in tmux session 'training'"
    echo ""
    echo "To monitor training:"
    echo "  ./deploy_gcp.sh monitor"
    echo ""
    echo "To download results:"
    echo "  ./deploy_gcp.sh download"
    echo ""
    echo "To stop VM:"
    echo "  ./deploy_gcp.sh stop"
}

# Function to monitor training
monitor_training() {
    echo -e "${GREEN}Connecting to training session...${NC}"
    echo "Press Ctrl+B then D to detach without stopping training"
    sleep 2
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID -- -t "tmux attach -t training"
}

# Function to check status
check_status() {
    echo -e "${GREEN}Checking training status...${NC}"
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID --command="
        if tmux has-session -t training 2>/dev/null; then
            echo '✓ Training session is running'
            tail -20 ~/mixture-of-thoughts/training.log
        else
            echo '✗ Training session not found'
        fi
    "
}

# Function to download results
download_results() {
    echo -e "${GREEN}Downloading results...${NC}"
    
    mkdir -p outputs
    gcloud compute scp --recurse $VM_NAME:~/mixture-of-thoughts/outputs/ ./outputs/ --zone=$ZONE --project=$PROJECT_ID
    gcloud compute scp $VM_NAME:~/mixture-of-thoughts/training.log ./outputs/ --zone=$ZONE --project=$PROJECT_ID || true
    
    echo -e "${GREEN}✓ Results downloaded to ./outputs/${NC}"
}

# Function to stop VM
stop_vm() {
    echo -e "${YELLOW}Stopping VM...${NC}"
    gcloud compute instances stop $VM_NAME --zone=$ZONE --project=$PROJECT_ID
    echo -e "${GREEN}✓ VM stopped${NC}"
    echo ""
    echo "To resume training later:"
    echo "  gcloud compute instances start $VM_NAME --zone=$ZONE"
}

# Function to delete VM
delete_vm() {
    echo -e "${RED}Deleting VM...${NC}"
    read -p "Are you sure you want to delete $VM_NAME? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        gcloud compute instances delete $VM_NAME --zone=$ZONE --project=$PROJECT_ID --quiet
        echo -e "${GREEN}✓ VM deleted${NC}"
    else
        echo "Cancelled"
    fi
}

# Function to SSH into VM
ssh_vm() {
    echo -e "${GREEN}Connecting to VM...${NC}"
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT_ID
}

# Function to show costs
show_costs() {
    echo "Cost Estimates (per hour):"
    echo ""
    echo "GPU Options:"
    echo "  nvidia-tesla-t4:  ~\$0.35/hr  (16GB, good for experimentation)"
    echo "  nvidia-tesla-v100: ~\$2.50/hr  (16GB, 3x faster training)"
    echo "  nvidia-tesla-a100: ~\$3.67/hr  (40GB, 5x faster training)"
    echo ""
    echo "Current configuration: $GPU_TYPE"
    echo ""
    echo "Estimated training time:"
    echo "  Small model (4 layers):   ~1 hour   (\$0.35)"
    echo "  Medium model (12 layers): ~4 hours  (\$1.40)"
    echo "  Large model (24 layers):  ~12 hours (\$4.20)"
    echo ""
    echo "⚠️  Remember to stop/delete the VM when done to avoid charges!"
}

# Main command handler
case "${1:-deploy}" in
    deploy)
        echo "Starting full deployment..."
        create_vm
        setup_vm
        upload_code
        start_training
        echo ""
        echo -e "${GREEN}========================================================================"
        echo "✅ Deployment complete!"
        echo "========================================================================${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Monitor training:    ./deploy_gcp.sh monitor"
        echo "  2. Check status:        ./deploy_gcp.sh status"
        echo "  3. Download results:    ./deploy_gcp.sh download"
        echo "  4. Stop VM:             ./deploy_gcp.sh stop"
        echo ""
        ;;
    
    create)
        create_vm
        ;;
    
    setup)
        setup_vm
        upload_code
        ;;
    
    upload)
        upload_code
        ;;
    
    train)
        start_training
        ;;
    
    monitor)
        monitor_training
        ;;
    
    status)
        check_status
        ;;
    
    download)
        download_results
        ;;
    
    stop)
        stop_vm
        ;;
    
    delete)
        delete_vm
        ;;
    
    ssh)
        ssh_vm
        ;;
    
    costs)
        show_costs
        ;;
    
    *)
        echo "Usage: $0 {deploy|create|setup|upload|train|monitor|status|download|stop|delete|ssh|costs}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full deployment (create + setup + upload + train)"
        echo "  create   - Create VM only"
        echo "  setup    - Setup environment on VM"
        echo "  upload   - Upload code to VM"
        echo "  train    - Start training"
        echo "  monitor  - Attach to training session"
        echo "  status   - Check training status"
        echo "  download - Download results from VM"
        echo "  stop     - Stop VM (preserves data)"
        echo "  delete   - Delete VM permanently"
        echo "  ssh      - SSH into VM"
        echo "  costs    - Show cost estimates"
        echo ""
        echo "Environment variables:"
        echo "  GCP_PROJECT_ID  - Your GCP project ID"
        echo "  GCP_ZONE        - Zone (default: us-central1-a)"
        echo "  VM_NAME         - VM name (default: auto-generated)"
        echo "  GPU_TYPE        - GPU type (default: nvidia-tesla-t4)"
        echo "  TRAIN_EPOCHS    - Training epochs (default: 20)"
        echo ""
        echo "Example with GPU configuration:"
        echo "  GPU_TYPE=nvidia-tesla-v100 ./deploy_gcp.sh deploy"
        exit 1
        ;;
esac
