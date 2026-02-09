#!/bin/bash
# Script to check CUDA environment and diagnose GPU issues

echo "=== CUDA Environment Check ==="

# Check CUDA environment variables
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check for CUDA installation
echo -e "\n=== CUDA Installation Check ==="
if command -v nvcc &> /dev/null; then
    echo "nvcc found: $(which nvcc)"
    nvcc --version
else
    echo "nvcc not found in PATH"
fi

# Check for CUDA libraries
echo -e "\n=== CUDA Libraries Check ==="
cuda_libs=("libcuda.so" "libcudart.so" "libcublas.so" "libcudnn.so")
for lib in "${cuda_libs[@]}"; do
    if ldconfig -p | grep -q "$lib"; then
        echo "✓ $lib found"
    else
        echo "✗ $lib not found"
    fi
done

# Check GPU devices
echo -e "\n=== GPU Device Check ==="
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi found:"
    nvidia-smi --list-gpus
else
    echo "nvidia-smi not found"
fi

# Check loaded modules
echo -e "\n=== Loaded Modules ==="
if command -v module &> /dev/null; then
    module list 2>/dev/null || echo "module command not available"
else
    echo "module command not found"
fi

# Check for common CUDA paths
echo -e "\n=== Common CUDA Paths ==="
cuda_paths=("/usr/local/cuda" "/opt/cuda" "/usr/cuda" "$CUDA_HOME")
for path in "${cuda_paths[@]}"; do
    if [ -n "$path" ] && [ -d "$path" ]; then
        echo "✓ CUDA path exists: $path"
        if [ -f "$path/lib64/libcuda.so" ]; then
            echo "  ✓ libcuda.so found"
        else
            echo "  ✗ libcuda.so not found"
        fi
    fi
done

echo -e "\n=== Recommendations ==="
echo "1. Load CUDA module: module load cuda"
echo "2. Load cuDNN module: module load cudnn"
echo "3. Set CUDA_HOME: export CUDA_HOME=/path/to/cuda"
echo "4. Update LD_LIBRARY_PATH: export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" 