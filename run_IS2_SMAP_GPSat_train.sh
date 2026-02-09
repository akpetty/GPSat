#!/bin/bash

# Script to run IS2_SMAP_GPSat_train.py with SMAP data

# Set parameters
NUM_DAYS="15"  # Number of days before/after target date
TEST_NAME="test_smap"  # Test name for output directory
USE_SMAP="true"  # Whether to use SMAP data (true/false)

# Run the training script
echo "Running IS2_SMAP_GPSat_train.py with parameters:"
echo "  - Number of days: $NUM_DAYS"
echo "  - Test name: $TEST_NAME"
echo "  - Use SMAP: $USE_SMAP"
echo ""

python IS2_SMAP_GPSat_train.py \
    --num_days $NUM_DAYS \
    --name $TEST_NAME \
    --smap $USE_SMAP

echo ""
echo "Training completed!" 