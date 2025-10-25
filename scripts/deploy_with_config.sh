#!/bin/bash
# Deploy GCP VM using config files (VM + Training configs separate)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Show usage
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <vm_config> [training_config] [command]"
    echo ""
    echo "Arguments:"
    echo "  vm_config       - VM configuration file (e.g. configs/vm/h100.yaml)"
    echo "  training_config - Training configuration file (optional, e.g. configs/training/large_multigpu.yaml)"
    echo "  command         - Deployment command (optional, default: deploy)"
    echo ""
    echo "Available VM configs:"
    ls -1 configs/vm/*.yaml 2>/dev/null | sed 's/^/  /'
    echo ""
    echo "Available training configs:"
    ls -1 configs/training/*.yaml 2>/dev/null | sed 's/^/  /'
    echo ""
    echo "Examples:"
    echo "  # Deploy H100 with large multi-GPU training:"
    echo "  $0 configs/vm/h100_multi.yaml configs/training/large_multigpu.yaml"
    echo ""
    echo "  # Deploy A100 with small test model:"
    echo "  $0 configs/vm/a100.yaml configs/training/small.yaml"
    echo ""
    echo "  # Deploy T4 with H100 training config (for testing large config on small GPU):"
    echo "  $0 configs/vm/t4.yaml configs/training/h100.yaml"
    exit 0
fi

# Get VM config file
VM_CONFIG="${1:-configs/vm/t4.yaml}"

if [ ! -f "$VM_CONFIG" ]; then
    echo -e "${RED}Error: VM config not found: $VM_CONFIG${NC}"
    echo ""
    echo "Available VM configs:"
    ls -1 configs/vm/*.yaml 2>/dev/null || echo "  No configs found"
    echo ""
    echo "Usage: $0 <vm_config> [training_config] [command]"
    exit 1
fi

# Get training config (optional)
TRAINING_CONFIG="${2:-}"
COMMAND="${3:-deploy}"

# If second arg looks like a command, use it as command and set default training config
if [ "$TRAINING_CONFIG" = "deploy" ] || [ "$TRAINING_CONFIG" = "monitor" ] || [ "$TRAINING_CONFIG" = "status" ] || [ "$TRAINING_CONFIG" = "download" ] || [ "$TRAINING_CONFIG" = "stop" ] || [ "$TRAINING_CONFIG" = "delete" ] || [ "$TRAINING_CONFIG" = "ssh" ] || [ "$TRAINING_CONFIG" = "costs" ]; then
    COMMAND="$TRAINING_CONFIG"
    TRAINING_CONFIG=""
fi

echo "========================================================================="
echo "🚀 GCP VM Deployment with Config"
echo "========================================================================="
echo "VM Config: $VM_CONFIG"
if [ -n "$TRAINING_CONFIG" ]; then
    echo "Training Config: $TRAINING_CONFIG"
fi
echo ""

# Parse YAML config (simple parsing)
parse_yaml() {
    grep "^$1:" "$2" | cut -d':' -f2 | tr -d ' '
}

# Read VM config
VM_NAME=$(parse_yaml "name" "$VM_CONFIG")
MACHINE_TYPE=$(parse_yaml "machine_type" "$VM_CONFIG")
GPU_TYPE=$(parse_yaml "gpu_type" "$VM_CONFIG")
GPU_COUNT=$(parse_yaml "gpu_count" "$VM_CONFIG")
ZONE=$(parse_yaml "zone" "$VM_CONFIG")
BOOT_DISK_SIZE=$(parse_yaml "boot_disk_size" "$VM_CONFIG")
DEFAULT_TRAINING_CONFIG=$(parse_yaml "training_config" "$VM_CONFIG")

echo "VM Configuration:"
echo "  VM Name: $VM_NAME"
echo "  Machine: $MACHINE_TYPE"
echo "  GPU: $GPU_TYPE x$GPU_COUNT"
echo "  Zone: $ZONE"
echo "  Disk: ${BOOT_DISK_SIZE}GB"

# If training config not specified on command line, try VM config default
if [ -z "$TRAINING_CONFIG" ]; then
    if [ -n "$DEFAULT_TRAINING_CONFIG" ]; then
        TRAINING_CONFIG="$DEFAULT_TRAINING_CONFIG"
        echo ""
        echo -e "${YELLOW}Using default training config from VM config: $TRAINING_CONFIG${NC}"
    else
        # Infer based on GPU type/count
        if [ "$GPU_COUNT" -gt 1 ]; then
            TRAINING_CONFIG="configs/training/large_multigpu.yaml"
            echo ""
            echo -e "${YELLOW}No training config specified. Inferring from GPU count: $TRAINING_CONFIG${NC}"
        else
            # Try to match GPU type to config
            case "$GPU_TYPE" in
                *h100*)
                    TRAINING_CONFIG="configs/training/h100.yaml"
                    ;;\
                *a100*)
                    TRAINING_CONFIG="configs/training/a100.yaml"
                    ;;\
                *l4*)
                    TRAINING_CONFIG="configs/training/l4.yaml"
                    ;;\
                *t4*)
                    TRAINING_CONFIG="configs/training/small.yaml"
                    ;;\
                *)
                    TRAINING_CONFIG="configs/training/medium.yaml"
                    ;;\
            esac
            echo ""
            echo -e "${YELLOW}No training config specified. Inferring from GPU type: $TRAINING_CONFIG${NC}"
        fi
    fi
fi

echo ""
echo "Training Config: $TRAINING_CONFIG"
echo ""

# Export as environment variables for deploy_gcp.sh
export VM_NAME
export MACHINE_TYPE
export GPU_TYPE
export GPU_COUNT
export ZONE
export TRAINING_CONFIG

# Run deployment
./scripts/deploy_gcp.sh "$COMMAND"
