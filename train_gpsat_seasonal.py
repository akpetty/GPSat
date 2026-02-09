###!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from datetime import datetime
import xarray as xr
import matplotlib.pyplot as plt
from GPSat import get_data_path
from GPSat.dataloader import DataLoader
from GPSat.models import GPModel
from GPSat.utils import get_config_from_yaml
import yaml

# Configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
if not os.path.exists(config_path):
    config = {
        'data_dir': '/path/to/your/data',
        'output_dir': 'output',
        'years': [2019, 2020, 2021],
        'months': [1, 2, 3, 4],
        'model_params': {
            'kernel': 'RBF',
            'lengthscale': 1.0,
            'variance': 1.0,
            'noise': 0.1
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
else:
    config = get_config_from_yaml(config_path)

# Create output directory if it doesn't exist
os.makedirs(config['output_dir'], exist_ok=True)

# Load data for the entire season
data_list = []
for year in config['years']:
    for month in config['months']:
        # Adjust the file path according to your data structure
        file_path = os.path.join(config['data_dir'], f'data_{year}_{month:02d}.nc')
        if os.path.exists(file_path):
            data = xr.open_dataset(file_path)
            data_list.append(data)
        else:
            print(f"Warning: File {file_path} not found.")

if not data_list:
    raise ValueError("No data files found for the specified years and months.")

# Combine all data
combined_data = xr.concat(data_list, dim='time')

# Prepare data for GPSat
X = combined_data[['latitude', 'longitude']].values
y = combined_data['thickness'].values

# Train the GPSat model
model = GPModel(**config['model_params'])
model.fit(X, y)

# Save the trained model
model.save(os.path.join(config['output_dir'], 'gpsat_model.pkl'))

print("GPSat model trained and saved successfully.") 