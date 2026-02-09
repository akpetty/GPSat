#!/bin/bash
#SBATCH --job-name=IS2_GPSat_train
#SBATCH --output=IS2_GPSat_train_%j.out
#SBATCH --error=IS2_GPSat_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --account=your_account_here

# Load any necessary modules
module load python/3.9
module load conda

# Activate your conda environment
source activate your_environment_name_here

# Set working directory
cd /home/aapetty/nobackup_symlink/GitHub/GPSat

# Run the script with your desired parameters
# Example: 15 days, test name "test1", with SIC data
python IS2_GPSat_train.py --num_days 15 --name test1 --sic true

# Or without SIC data:
# python IS2_GPSat_train.py --num_days 15 --name test1 --sic false 