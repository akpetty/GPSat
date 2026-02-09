#!/bin/bash
# Script to set up CUDA environment for TensorFlow GPU usage

echo "=== Setting up CUDA Environment ==="

# Check if we're in a SLURM environment
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running in SLURM environment"
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
fi

# Try to load CUDA modules
echo -e "\n=== Loading CUDA Modules ==="

# Common module names for CUDA
cuda_modules=("cuda" "cuda/11.8" "cuda/12.0" "cuda/12.1" "cuda/12.2")
cudnn_modules=("cudnn" "cudnn/8.6" "cudnn/8.9" "cudnn/9.0")

# Try to load CUDA module
cuda_loaded=false
for module in "${cuda_modules[@]}"; do
    if module load "$module" 2>/dev/null; then
        echo "✓ Loaded CUDA module: $module"
        cuda_loaded=true
        break
    fi
done

if [ "$cuda_loaded" = false ]; then
    echo "⚠ Could not load CUDA module automatically"
    echo "Available modules:"
    module avail cuda 2>/dev/null || echo "No cuda modules found"
fi

# Try to load cuDNN module
cudnn_loaded=false
for module in "${cudnn_modules[@]}"; do
    if module load "$module" 2>/dev/null; then
        echo "✓ Loaded cuDNN module: $module"
        cudnn_loaded=true
        break
    fi
done

if [ "$cudnn_loaded" = false ]; then
    echo "⚠ Could not load cuDNN module automatically"
    echo "Available modules:"
    module avail cudnn 2>/dev/null || echo "No cudnn modules found"
fi

# Set environment variables
echo -e "\n=== Setting Environment Variables ==="

# Find CUDA installation
cuda_path=""
for path in "/usr/local/cuda" "/opt/cuda" "/usr/cuda" "$CUDA_HOME"; do
    if [ -n "$path" ] && [ -d "$path" ]; then
        cuda_path="$path"
        break
    fi
done

if [ -n "$cuda_path" ]; then
    echo "Found CUDA at: $cuda_path"
    export CUDA_HOME="$cuda_path"
    export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
    echo "Set CUDA_HOME=$CUDA_HOME"
    echo "Updated LD_LIBRARY_PATH"
else
    echo "⚠ CUDA installation not found in common locations"
fi

# Check if CUDA is working
echo -e "\n=== Testing CUDA Setup ==="
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found: $(which nvcc)"
    nvcc --version | head -1
else
    echo "✗ nvcc not found"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found: $(which nvidia-smi)"
    nvidia-smi --list-gpus | head -5
else
    echo "✗ nvidia-smi not found"
fi

echo -e "\n=== Environment Summary ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo -e "\n=== Next Steps ==="
echo "1. Run: python test_gpu.py"
echo "2. If GPU is still not detected, try:"
echo "   - module load cuda/11.8"
echo "   - module load cudnn/8.6"
echo "   - export CUDA_HOME=/usr/local/cuda"
echo "3. Then run your GPSat script again" 