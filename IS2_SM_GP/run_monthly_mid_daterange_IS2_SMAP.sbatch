#!/bin/bash
#SBATCH --job-name=IS2_SMAP_monthly_mid
#SBATCH --output=slurm_output/IS2_SMAP_monthly_mid_%A_%a.out
#SBATCH --error=slurm_output/IS2_SMAP_monthly_mid_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akpetty@umd.edu
#SBATCH --array=0-47%4  # Up to 4 nodes in parallel; set 0-N%4 to match DATES length (e.g. 0-47%4 for DATES_FULL)

# Usage:
# Processes the 15th (mid-month) of various months across a date range.
# Runs up to 4 array tasks in parallel (4 nodes). Edit #SBATCH --array to match DATES length (e.g. 0-47%4 for 48 dates).
# Edit the DATES array below with your desired dates, then:
# sbatch run_monthly_mid_daterange_IS2_SMAP.sbatch NUM_DAYS SMAP DATASET_VERSION [VARIABLE] [IS2_PREF] [SMOOTHED_PARAMS_FILE]
# Example:
# sbatch run_monthly_mid_daterange_IS2_SMAP.sbatch 30 true dev_v1
# sbatch run_monthly_mid_daterange_IS2_SMAP.sbatch 30 true dev_v1 thickness
# Note: VARIABLE defaults to "thickness"; IS2_PREF defaults to "true" (use 5th arg or IS2_PREF=false to disable).

set -eo pipefail

# Create slurm_output directory if it doesn't exist
mkdir -p slurm_output

# ========================================
# EDIT THIS: List of dates to process (YYYY-MM-DD format)
# ========================================
#DATES=(
#    "2019-01-15"
#    "2025-04-15"
#)
# Full date range (uncomment and set #SBATCH --array=0-47%4 to use):
DATES=(
     "2018-11-15" "2018-12-15" "2019-01-15" "2019-02-15" "2019-03-15" "2019-04-15"
     "2019-09-15" "2019-10-15" "2019-11-15" "2019-12-15"
     "2020-01-15" "2020-02-15" "2020-03-15" "2020-04-15"
     "2020-09-15" "2020-10-15" "2020-11-15" "2020-12-15"
     "2021-01-15" "2021-02-15" "2021-03-15" "2021-04-15"
     "2021-09-15" "2021-10-15" "2021-11-15" "2021-12-15"
     "2022-01-15" "2022-02-15" "2022-03-15" "2022-04-15"
     "2022-09-15" "2022-10-15" "2022-11-15" "2022-12-15"
     "2023-01-15" "2023-02-15" "2023-03-15" "2023-04-15"
     "2023-09-15" "2023-10-15" "2023-11-15" "2023-12-15"
     "2024-01-15" "2024-02-15" "2024-03-15" "2024-04-15"
     "2024-09-15" "2024-10-15" "2024-11-15" "2024-12-15"
     "2025-01-15" "2025-02-15" "2025-03-15" "2025-04-15"
)
#)
# ========================================
#
# GP default parameters (IS2_SMAP_GPSat_train.py; anti_streak-style defaults):
#   expert_spacing: 300_000 m
#   noise_std: 0.3
#   jitter: 1e-3
#   lengthscale bounds: low [10000, 10000, 1.0] (10 km for x,y; 1 day for t), high [500000, 500000, 50]
#   hyperparameter smoothing: l_x, l_y = 300_000 m; kernel_variance max = 2.5
#

# Output path - can be overridden with OUTPUT_PATH environment variable
OUTPUT_PATH="${OUTPUT_PATH:-/explore/nobackup/people/aapetty/GitHub/GPSat/output/}"
# Ensure it ends with /
if [[ ! "$OUTPUT_PATH" =~ /$ ]]; then
    OUTPUT_PATH="${OUTPUT_PATH}/"
fi

# Parse arguments
NUM_DAYS="$1"          # window days
SMAP_FLAG="$2"         # true/false
DATASET_VERSION="$3"   # dataset version string (e.g., dev_v1, v1.0)
VARIABLE="${4:-thickness}"  # variable name (default: thickness)
# IS2_PREF: 5th arg, or env IS2_PREF; default true
IS2_PREF="${5:-${IS2_PREF:-true}}"
SMOOTHED_PARAMS_ARG="${6:-}"  # optional smoothed params file from command line

if [ -z "$NUM_DAYS" ] || [ -z "$SMAP_FLAG" ] || [ -z "$DATASET_VERSION" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: sbatch $0 NUM_DAYS SMAP DATASET_VERSION [VARIABLE] [IS2_PREF] [SMOOTHED_PARAMS_FILE]"
    echo "  IS2_PREF defaults to true; set 5th arg or IS2_PREF=false to disable."
    exit 1
fi

# Name is always "run" - hardcoded
NAME="run"

# --- Set Smoothed Parameters Path ---
if [ -n "$SMOOTHED_PARAMS_ARG" ]; then
    SMOOTHED_PARAMS="$SMOOTHED_PARAMS_ARG"
else
    # If not provided, construct the default path based on the 15th of the same month as TARGET_DATE
    # (We'll set this after we know TARGET_DATE, but prepare the logic here)
    SMOOTHED_PARAMS=""  # Will be set below after TARGET_DATE is determined
fi

# Environment setup
module purge
module load gcc/12.1.0 2>/dev/null || true

export PATH="/home/aapetty/.conda/envs/is2_39_gp/bin:$PATH"
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda &> /dev/null; then
  conda activate is2_39_gp || source activate is2_39_gp
fi

# Get the date for this array task
if [ ${SLURM_ARRAY_TASK_ID} -ge ${#DATES[@]} ]; then
    echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} exceeds date array size ${#DATES[@]}"
    exit 0
fi

TARGET_DATE="${DATES[$SLURM_ARRAY_TASK_ID]}"

# If smoothed params not provided, construct path based on 15th of same month
if [ -z "$SMOOTHED_PARAMS_ARG" ]; then
    # Extract year and month from TARGET_DATE
    YEAR=$(echo "$TARGET_DATE" | cut -d'-' -f1)
    MONTH=$(echo "$TARGET_DATE" | cut -d'-' -f2)
    DATE_15TH_NODASH="${YEAR}${MONTH}15"
    
    SMAP_DIR_PART=""
    if [ "$SMAP_FLAG" == "true" ]; then
        SMAP_DIR_PART="_smap"
    fi
    
    # Look in the new structure: {OUTPUT_PATH}/{DATASET_VERSION}/{VARIABLE}/run_{NUM_DAYS}days{SMAP_DIR_PART}_{DATE}_v{NN}/
    SEARCH_DIR="${OUTPUT_PATH}/${DATASET_VERSION}/${VARIABLE}"
    PREFIX="run_${NUM_DAYS}days${SMAP_DIR_PART}_${DATE_15TH_NODASH}_v"
    
    SMOOTHED_PARAMS_DIR=""
    if [ -d "$SEARCH_DIR" ]; then
        # Find all directories matching the pattern and get the latest version
        MATCHING_DIRS=$(find "$SEARCH_DIR" -maxdepth 1 -type d -name "${PREFIX}*" 2>/dev/null | sort -V | tail -1)
        if [ -n "$MATCHING_DIRS" ]; then
            SMOOTHED_PARAMS_DIR="$MATCHING_DIRS"
            echo "INFO: Found smoothed params directory: $(basename "$SMOOTHED_PARAMS_DIR")"
        fi
    fi
    
    # If not found, try old structure for backward compatibility
    if [ -z "$SMOOTHED_PARAMS_DIR" ] || [ ! -d "$SMOOTHED_PARAMS_DIR" ]; then
        OLD_SEARCH_DIR="${OUTPUT_PATH}"
        if [ -d "$OLD_SEARCH_DIR" ]; then
            MATCHING_DIRS=$(find "$OLD_SEARCH_DIR" -maxdepth 1 -type d -name "${PREFIX}*" 2>/dev/null | sort -V | tail -1)
            if [ -n "$MATCHING_DIRS" ]; then
                SMOOTHED_PARAMS_DIR="$MATCHING_DIRS"
                echo "INFO: Found smoothed params in old structure: $(basename "$SMOOTHED_PARAMS_DIR")"
            fi
        fi
    fi
    
    
    # If directory doesn't exist, try without the NAME suffix (e.g., multiparams_feb -> multiparams)
    if [ ! -d "$SMOOTHED_PARAMS_DIR" ]; then
        # Extract base name (remove potential suffix like _feb)
        BASE_NAME=$(echo "$NAME" | sed 's/_[^_]*$//')
        if [ "$BASE_NAME" != "$NAME" ] && [ -n "$BASE_NAME" ]; then
            ALT_DIR_NAME="${BASE_NAME}_${NUM_DAYS}days${SMAP_DIR_PART}_${DATE_15TH_NODASH}_v01"
            ALT_SMOOTHED_PARAMS_DIR="${OUTPUT_PATH}/${DATASET_VERSION}/${VARIABLE}/${ALT_DIR_NAME}"
            if [ -d "$ALT_SMOOTHED_PARAMS_DIR" ]; then
                SMOOTHED_PARAMS_DIR="$ALT_SMOOTHED_PARAMS_DIR"
                echo "INFO: Using alternate directory: ${ALT_DIR_NAME}"
            fi
        fi
        # Also try old structure (without dataset_version/variable) for backward compatibility
        if [ ! -d "$SMOOTHED_PARAMS_DIR" ]; then
            OLD_STRUCTURE_DIR="${OUTPUT_PATH}/${DEFAULT_DIR_NAME}"
            if [ -d "$OLD_STRUCTURE_DIR" ]; then
                SMOOTHED_PARAMS_DIR="$OLD_STRUCTURE_DIR"
                echo "INFO: Using old directory structure: ${OLD_STRUCTURE_DIR}"
            fi
        fi
    fi
    
    # Check if directory exists before looking for files
    if [ -z "$SMOOTHED_PARAMS_DIR" ] || [ ! -d "$SMOOTHED_PARAMS_DIR" ]; then
        echo "WARNING: Could not find smoothed params directory for 15th of month"
        echo "  Searched for: ${SEARCH_DIR}/${PREFIX}*"
        echo "  Matching: version=${DATASET_VERSION}, variable=${VARIABLE}, num_days=${NUM_DAYS}, smap=${SMAP_FLAG}"
        SMOOTHED_PARAMS=""
    else
        # Look for H5 file in the directory - must match the expected filename
        H5_FILENAME="${H5_FILENAME:-IS2_interp_test_petty_v1.h5}"
        H5_FILE="${SMOOTHED_PARAMS_DIR}/${H5_FILENAME}"
        if [ -f "$H5_FILE" ]; then
            SMOOTHED_PARAMS="$H5_FILE"
            echo "INFO: Using smoothed params from: ${SMOOTHED_PARAMS}"
        else
            echo "WARNING: Could not find expected H5 file: ${H5_FILENAME} in ${SMOOTHED_PARAMS_DIR}"
            echo "  Expected file: ${H5_FILE}"
            SMOOTHED_PARAMS=""
        fi
    fi
fi

echo "=========================================="
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Processing date: ${TARGET_DATE}"
echo "Total dates: ${#DATES[@]}"
echo "Num days: ${NUM_DAYS}"
echo "SMAP: ${SMAP_FLAG}"
echo "Dataset version: ${DATASET_VERSION}"
echo "Variable: ${VARIABLE}"
echo "IS2 pref: ${IS2_PREF}"
echo "Smoothed params: ${SMOOTHED_PARAMS}"
echo "Output path: ${OUTPUT_PATH}"
echo "=========================================="

# Python script: use absolute path (edit if your repo lives elsewhere)
PYTHON_SCRIPT="${PYTHON_SCRIPT:-/explore/nobackup/people/aapetty/GitHub/GPSat/IS2_SMAP_GPSat_train.py}"

# Build command (CA excluded and smap_exclude_regions are on by default in Python script)
CMD=(python "${PYTHON_SCRIPT}" \
  --num_days "${NUM_DAYS}" \
  --smap "${SMAP_FLAG}" \
  --target_date "${TARGET_DATE}" \
  --output_results_path "${OUTPUT_PATH}" \
  --dataset_version "${DATASET_VERSION}" \
  --variable "${VARIABLE}" \
  --is2_pref "${IS2_PREF}")

if [ -n "${SMOOTHED_PARAMS}" ]; then
    CMD+=(--smoothed_params_file_path "${SMOOTHED_PARAMS}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Completed date: ${TARGET_DATE}"
