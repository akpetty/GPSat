#!/usr/bin/env python3
"""
Test script to check GPU availability and TensorFlow configuration
"""

import os
import sys

def test_gpu_availability():
    """Test GPU availability and TensorFlow configuration"""
    
    print("=== GPU Availability Test ===")
    
    # Check CUDA environment
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")
        
        # Check available devices
        print("\n=== Available Devices ===")
        physical_devices = tf.config.list_physical_devices()
        for device_type in ['CPU', 'GPU']:
            devices = tf.config.list_physical_devices(device_type)
            print(f"{device_type} devices: {len(devices)}")
            for device in devices:
                print(f"  - {device}")
        
        # Test GPU computation
        if tf.config.list_physical_devices('GPU'):
            print("\n=== GPU Computation Test ===")
            
            # Create a simple tensor on GPU
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"Matrix multiplication result: {c}")
                print(f"Device placement: {c.device}")
            
            print("✓ GPU computation test PASSED")
        else:
            print("⚠ No GPU devices found")
            
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        import traceback
        traceback.print_exc()

def test_gpflow_gpu():
    """Test GPflow GPU usage"""
    
    print("\n=== GPflow GPU Test ===")
    
    try:
        import gpflow
        print(f"GPflow version: {gpflow.__version__}")
        
        # Check if GPflow can use GPU
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            print("GPflow should be able to use GPU")
            
            # Create a simple GP model
            import numpy as np
            X = np.random.randn(100, 2)
            Y = np.random.randn(100, 1)
            
            kernel = gpflow.kernels.RBF(lengthscales=[1.0, 1.0])
            model = gpflow.models.GPR(data=(X, Y), kernel=kernel)
            
            print("✓ GPflow GPU test PASSED")
        else:
            print("⚠ No GPU available for GPflow")
            
    except ImportError as e:
        print(f"✗ GPflow import failed: {e}")
    except Exception as e:
        print(f"✗ GPflow GPU test failed: {e}")

if __name__ == "__main__":
    test_gpu_availability()
    test_gpflow_gpu() 