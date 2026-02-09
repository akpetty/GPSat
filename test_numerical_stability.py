#!/usr/bin/env python3
"""
Test script for numerical stability with small dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

def create_test_data():
    """Create a small test dataset to check numerical stability"""
    
    print("Creating test dataset...")
    
    # Create a small grid of points
    x_coords = np.linspace(-1000000, 1000000, 20)
    y_coords = np.linspace(-1000000, 1000000, 20)
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    
    # Create some test thickness values
    thickness = 0.5 + 0.3 * np.sin(x_mesh/500000) * np.cos(y_mesh/500000)
    
    # Add some noise
    np.random.seed(42)
    thickness += np.random.normal(0, 0.1, thickness.shape)
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'x': x_mesh.flatten(),
        'y': y_mesh.flatten(),
        'ice_thickness': thickness.flatten(),
        'time': pd.to_datetime('2019-04-15')
    })
    
    # Add small random noise to prevent exact duplicates
    test_data['x'] += np.random.normal(0, 1e-6, len(test_data))
    test_data['y'] += np.random.normal(0, 1e-6, len(test_data))
    
    print(f"Test dataset created: {len(test_data)} points")
    print(f"Thickness range: {test_data['ice_thickness'].min():.3f} to {test_data['ice_thickness'].max():.3f}")
    
    return test_data

def test_gpflow_stability():
    """Test GPflow with the test dataset"""
    
    try:
        import gpflow
        print("GPflow imported successfully")
        
        # Create test data
        test_data = create_test_data()
        
        # Create a simple GP model
        X = test_data[['x', 'y']].values
        Y = test_data['ice_thickness'].values.reshape(-1, 1)
        
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        
        # Create kernel with increased jitter
        kernel = gpflow.kernels.RBF(lengthscales=[50000, 50000])
        
        # Create model with increased jitter
        model = gpflow.models.GPR(
            data=(X, Y),
            kernel=kernel,
            noise_variance=0.1,
            mean_function=None
        )
        
        # Set jitter
        gpflow.config.set_default_jitter(1e-4)
        
        print("GP model created successfully")
        
        # Try to optimize
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        
        print("GP optimization completed successfully")
        print("✓ Numerical stability test PASSED")
        
    except Exception as e:
        print(f"✗ Numerical stability test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpflow_stability() 