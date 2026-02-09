#!/bin/bash
# Usage: sbatch submit_IS2_SMAP.sh <num_days> <name> <smap_flag> <target_date>
# Example: sbatch submit_IS2_SMAP.sh 0 test_smap True 2019-04-05
#
# Debug locally (outside SLURM) after activating env:
#   module load gcc/12.1.0
#   source "$HOME/.conda/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniconda3/etc/profile.d/conda.sh"
#   conda activate is2_39_gp
#   python IS2_SMAP_GPSat_train.py --num_days 0 --name test_smap --smap True --target_date 2019-04-05
#
# Quick dry-run (prints help):
#   python IS2_SMAP_GPSat_train.py --help

#SBATCH --job-name=is2_smap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=is2_smap_%j.out
#SBATCH --error=is2_smap_%j.err
##SBATCH --mail-user=akpetty@umd.edu
##SBATCH --mail-type=END,FAIL

set -euo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: sbatch submit_IS2_SMAP.sh <num_days> <name> <smap_flag> <target_date YYYY-MM-DD>" >&2
  exit 1
fi

NUM_DAYS="$1"
NAME="$2"
SMAP_FLAG="$3"
TARGET_DATE="$4"

# Conda activation (robust)
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
module load gcc/12.1.0 || true
conda activate is2_39_gp || { echo "Failed to activate env is2_39_gp" >&2; exit 2; }

PYTHON=python
SCRIPT="$(pwd)/IS2_SMAP_GPSat_train.py"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: Script not found at $SCRIPT" >&2
  exit 3
fi

echo "Running IS2_SMAP_GPSat_train.py with: num_days=$NUM_DAYS name=$NAME smap=$SMAP_FLAG target_date=$TARGET_DATE";
$PYTHON "$SCRIPT" --num_days "$NUM_DAYS" --name "$NAME" --smap "$SMAP_FLAG" --target_date "$TARGET_DATE" || { echo "Python run failed" >&2; exit 4; }

echo "Job complete."