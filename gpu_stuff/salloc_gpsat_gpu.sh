#!/bin/bash
# Interactive SLURM allocation for GPSat GPU training
# Run as: ./salloc_gpsat_gpu.sh

echo "Requesting interactive SLURM allocation..."

# Request GPU resources
salloc \
    --job-name=gpu_gpsat \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --partition=compute \
    --time=24:00:00

echo "Allocation complete. You can now run your GPSat training script."
echo "Example: python IS2_SMAP_GPSat_train.py --num_days 15 --name test_smap --smap true" 