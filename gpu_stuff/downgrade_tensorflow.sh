#!/bin/bash
# Script to downgrade TensorFlow to a more stable version

echo "=== Downgrading TensorFlow ==="
echo "Current version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

# Recommended stable versions
echo -e "\nRecommended stable TensorFlow versions:"
echo "1. TensorFlow 2.15.0 (very stable, good CUDA compatibility)"
echo "2. TensorFlow 2.16.0 (stable, recent features)"
echo "3. TensorFlow 2.14.0 (very stable, widely used)"

# Ask user for version choice
read -p "Enter TensorFlow version to install (e.g., 2.15.0): " tf_version

if [ -z "$tf_version" ]; then
    tf_version="2.15.0"
    echo "Using default version: $tf_version"
fi

echo -e "\n=== Installing TensorFlow $tf_version ==="

# Uninstall current TensorFlow
echo "Uninstalling current TensorFlow..."
pip uninstall -y tensorflow tensorflow-gpu

# Install specific version
echo "Installing TensorFlow $tf_version..."
pip install "tensorflow==$tf_version"

# Verify installation
echo -e "\n=== Verification ==="
python -c "import tensorflow as tf; print(f'New TensorFlow version: {tf.__version__}')"

# Test GPU detection
echo -e "\n=== Testing GPU Detection ==="
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'Available devices: {tf.config.list_physical_devices()}')
print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')
"

echo -e "\n=== Next Steps ==="
echo "1. Test GPU detection: python test_gpu.py"
echo "2. If GPU is detected, run your GPSat script"
echo "3. If issues persist, try TensorFlow 2.14.0" 