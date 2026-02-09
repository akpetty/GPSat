#!/bin/bash
#SBATCH --job-name=IS2_GPSat
#SBATCH --output=IS2_GPSat_%j.out
#SBATCH --error=IS2_GPSat_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=compute

# Change these to match your setup:
# 1. Replace 'your_environment_name' with your actual conda environment name
# 2. Adjust memory (--mem) if needed
# 3. Adjust time limit (--time) if needed

# Activate conda environment
source activate your_environment_name

# Run the script
cd /home/aapetty/nobackup_symlink/GitHub/GPSat
python IS2_GPSat_train.py --num_days 15 --name test1 --sic true 