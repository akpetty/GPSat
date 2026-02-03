#!/bin/bash
#SBATCH --job-name=IS2_SMAP_days_in_month
#SBATCH --output=slurm_output/IS2_SMAP_days_in_month_%A_%a.out
#SBATCH --error=slurm_output/IS2_SMAP_days_in_month_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akpetty@umd.edu
#SBATCH --array=0-29%4  # Max days in a month minus 1; %4 limits to 4 concurrent jobs.

# ==============================================================================
# Usage:
# Runs every day of a given month EXCEPT the 15th.
# sbatch run_days_in_month_IS2_SMAP.sbatch YYYY-MM NUM_DAYS SMAP_FLAG DATASET_VERSION [VARIABLE] [SMOOTHED_PARAMS_FILE]
#
# If SMOOTHED_PARAMS_FILE is omitted, it automatically finds and loads smoothed params
# from the 15th of the same month, matching:
#   - Same DATASET_VERSION
#   - Same VARIABLE (or default "thickness")
#   - Same NUM_DAYS
#   - Same SMAP_FLAG
# It will use the latest version (v01, v02, etc.) if multiple exist.
#
# Example (January 2020, all days except 15th, dev_v1 output):
# sbatch run_days_in_month_IS2_SMAP.sbatch 2020-01 30 true dev_v1
# sbatch run_days_in_month_IS2_SMAP.sbatch 2023-02 30 true dev_v1 thickness
# Note: VARIABLE defaults to "thickness" if not provided
# ==============================================================================

set -eo pipefail

# Create slurm_output directory if it doesn't exist
mkdir -p slurm_output

# --- Argument Parsing ---
TARGET_MONTH_STR="$1"
NUM_DAYS="$2"
SMAP_FLAG="$3"
DATASET_VERSION="$4"   # dataset version string (e.g., dev_v1, v1.0)
VARIABLE="${5:-thickness}"  # variable name (default: thickness)
SMOOTHED_PARAMS_ARG="${6:-}" # Optional smoothed params file from command line

if [ -z "$TARGET_MONTH_STR" ] || [ -z "$NUM_DAYS" ] || [ -z "$SMAP_FLAG" ] || [ -z "$DATASET_VERSION" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: sbatch $0 YYYY-MM NUM_DAYS SMAP_FLAG DATASET_VERSION [VARIABLE] [SMOOTHED_PARAMS_FILE]"
    exit 1
fi

# Name is always "run" - hardcoded
NAME="run"

# Output path - can be overridden with OUTPUT_PATH environment variable
OUTPUT_PATH="${OUTPUT_PATH:-/explore/nobackup/people/aapetty/GitHub/GPSat/output/}"
# Ensure it ends with /
if [[ ! "$OUTPUT_PATH" =~ /$ ]]; then
    OUTPUT_PATH="${OUTPUT_PATH}/"
fi
YEAR=$(echo "$TARGET_MONTH_STR" | cut -d'-' -f1)
MONTH=$(echo "$TARGET_MONTH_STR" | cut -d'-' -f2)

# --- Set Smoothed Parameters Path ---
if [ -n "$SMOOTHED_PARAMS_ARG" ]; then
    SMOOTHED_PARAMS="$SMOOTHED_PARAMS_ARG"
else
    # If not provided, automatically find smoothed params from the 15th of the same month
    # Matching: same DATASET_VERSION, same NAME, same VARIABLE, same NUM_DAYS, same SMAP_FLAG
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
    
    # Check if directory exists and look for H5 file
    if [ -z "$SMOOTHED_PARAMS_DIR" ] || [ ! -d "$SMOOTHED_PARAMS_DIR" ]; then
        echo "WARNING: Could not find smoothed params directory for 15th of month"
        echo "  Searched for: ${SEARCH_DIR}/${PREFIX}*"
        echo "  Matching: version=${DATASET_VERSION}, variable=${VARIABLE}, num_days=${NUM_DAYS}, smap=${SMAP_FLAG}"
        echo "  Will run without loading smoothed params (will optimize from scratch)"
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

# --- Date Generation (skipping the 15th) ---
DATES=()
DAYS_IN_MONTH=$(cal $MONTH $YEAR | awk 'NF {DAYS = $NF}; END {print DAYS}')

for ((day=1; day<=$DAYS_IN_MONTH; day++)); do
    if [ "$day" -ne 15 ]; then
        DATES+=("$(printf '%s-%s-%02d' "$YEAR" "$MONTH" "$day")")
    fi
done

# --- Environment Setup ---
module purge
module load gcc/12.1.0 2>/dev/null || true

export PATH="/home/aapetty/.conda/envs/is2_39_gp/bin:$PATH"
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda &> /dev/null; then
  conda activate is2_39_gp || source activate is2_39_gp
fi

# --- Job Execution ---
# Check if the array task ID is valid for the number of days in the month
if [ ${SLURM_ARRAY_TASK_ID} -ge ${#DATES[@]} ]; then
    echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} is out of bounds for month ${TARGET_MONTH_STR} which has ${#DATES[@]} days to process. Exiting."
    exit 0
fi

TARGET_DATE="${DATES[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Processing date: ${TARGET_DATE}"
echo "Num days (window): ${NUM_DAYS}"
echo "SMAP flag: ${SMAP_FLAG}"
echo "Dataset version: ${DATASET_VERSION}"
echo "Variable: ${VARIABLE}"
echo "Smoothed params file: ${SMOOTHED_PARAMS}"
echo "=========================================="

# Python script: use absolute path (edit if your repo lives elsewhere)
PYTHON_SCRIPT="${PYTHON_SCRIPT:-/explore/nobackup/people/aapetty/GitHub/GPSat/IS2_SMAP_GPSat_train.py}"

# Build and run the command
CMD=(python "${PYTHON_SCRIPT}" \
  --num_days "${NUM_DAYS}" \
  --smap "${SMAP_FLAG}" \
  --target_date "${TARGET_DATE}" \
  --output_results_path "${OUTPUT_PATH}" \
  --dataset_version "${DATASET_VERSION}" \
  --variable "${VARIABLE}")

if [ -n "${SMOOTHED_PARAMS}" ]; then
    CMD+=(--smoothed_params_file_path "${SMOOTHED_PARAMS}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Completed date: ${TARGET_DATE}"
