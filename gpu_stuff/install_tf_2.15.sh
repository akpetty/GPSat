#!/bin/bash
# Simple script to install TensorFlow 2.15.0

echo "=== Installing TensorFlow 2.15.0 ==="

# Uninstall current TensorFlow
echo "Uninstalling current TensorFlow..."
pip uninstall -y tensorflow tensorflow-gpu

# Install TensorFlow 2.15.0
echo "Installing TensorFlow 2.15.0..."
pip install "tensorflow==2.15.0"

# Verify installation
echo -e "\n=== Verification ==="
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

echo "Installation complete!" 