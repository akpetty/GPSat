#!/bin/bash
#SBATCH --job-name=IS2_SIC_TF_SD
#SBATCH --output=slurm_output/IS2_SIC_TF_SD_%A_%a.out
#SBATCH --error=slurm_output/IS2_SIC_TF_SD_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akpetty@umd.edu
#SBATCH --array=0-47%2  # Up to 2 nodes in parallel; set 0-N%4 to match DATES length (e.g. 0-47%4 for 48 dates)

# Usage:
# IS2 + SIC (no SMAP). SIC used as ice-edge guide (values 0 where SIC < 0.15).
# Predicts freeboard (ICESat-2 total freeboard) or snow depth (SD). VARIABLE: freeboard or snow_depth.
# NO "true"/SMAP argument here (unlike run_monthly_mid_daterange_IS2_SMAP.sbatch): args are NUM_DAYS DATASET_VERSION [VARIABLE] ...
# Runs up to 4 array tasks in parallel (4 nodes). Edit #SBATCH --array to match DATES length (e.g. 0-11%4 for 12 dates).
# sbatch run_monthly_mid_daterange_IS2_SIC_TF_SD.sbatch NUM_DAYS DATASET_VERSION [VARIABLE] [IS2_PREF] [SMOOTHED_PARAMS_FILE]
# Example:
# sbatch run_monthly_mid_daterange_IS2_SIC_TF_SD.sbatch 30 dev_v3 freeboard
# sbatch run_monthly_mid_daterange_IS2_SIC_TF_SD.sbatch 30 dev_v3 snow_depth
# Note: VARIABLE defaults to "freeboard"; IS2_PREF defaults to "true" (use 4th arg or IS2_PREF=false to disable).
# Without 5th arg: run optimizes from scratch. To auto-use a previous run's params, set AUTO_LOAD_SMOOTHED_PARAMS=true before sbatch.

set -eo pipefail

# Create slurm_output directory if it doesn't exist
mkdir -p slurm_output

# ========================================
#
# GP default parameters (IS2_SMAP_GPSat_train.py; anti_streak-style defaults):
#   expert_spacing: 300_000 m
#   noise_std: 0.3
#   jitter: 1e-3
#   lengthscale bounds: low [10000, 10000, 1.0] (10 km for x,y; 1 day for t), high [500000, 500000, 50]
#   hyperparameter smoothing: l_x, l_y = 300_000 m; kernel_variance max = 2.5
#
# ========================================
# EDIT THIS: List of dates to process (YYYY-MM-DD format)
# ========================================
#DATES=(
#    "2019-01-15"
#    "2025-04-15"
#)

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

# ========================================

# Output path - can be overridden with OUTPUT_PATH environment variable
OUTPUT_PATH="${OUTPUT_PATH:-/explore/nobackup/people/aapetty/GitHub/GPSat/output/}"
# Ensure it ends with /
if [[ ! "$OUTPUT_PATH" =~ /$ ]]; then
    OUTPUT_PATH="${OUTPUT_PATH}/"
fi

# Parse arguments: NUM_DAYS DATASET_VERSION VARIABLE [IS2_PREF] [SMOOTHED_PARAMS_FILE]
NUM_DAYS="$1"
DATASET_VERSION="$2"
VARIABLE="${3:-freeboard}"  # freeboard or snow_depth
# IS2_PREF: 4th arg, or env IS2_PREF; default true
IS2_PREF="${4:-${IS2_PREF:-true}}"
SMOOTHED_PARAMS_ARG="${5:-}"

# Only auto-discover smoothed params if explicitly requested (set AUTO_LOAD_SMOOTHED_PARAMS=true).
# Otherwise we optimize from scratch unless you pass a file as 5th argument.
AUTO_LOAD_SMOOTHED_PARAMS="${AUTO_LOAD_SMOOTHED_PARAMS:-false}"

if [ -z "$NUM_DAYS" ] || [ -z "$DATASET_VERSION" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: sbatch $0 NUM_DAYS DATASET_VERSION [VARIABLE] [IS2_PREF] [SMOOTHED_PARAMS_FILE]"
    echo "  IS2_PREF defaults to true; set 4th arg or IS2_PREF=false to disable."
    echo "  Without 5th arg: optimize from scratch. Set AUTO_LOAD_SMOOTHED_PARAMS=true to auto-use a previous run's params."
    exit 1
fi

NAME="run"

# --- Set Smoothed Parameters Path ---
if [ -n "$SMOOTHED_PARAMS_ARG" ]; then
    SMOOTHED_PARAMS="$SMOOTHED_PARAMS_ARG"
else
    SMOOTHED_PARAMS=""  # Set below only when AUTO_LOAD_SMOOTHED_PARAMS=true
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

# Smoothed params: only auto-discover when AUTO_LOAD_SMOOTHED_PARAMS=true (otherwise optimize from scratch)
if [ -z "$SMOOTHED_PARAMS_ARG" ] && [ "$AUTO_LOAD_SMOOTHED_PARAMS" = "true" ]; then
    YEAR=$(echo "$TARGET_DATE" | cut -d'-' -f1)
    MONTH=$(echo "$TARGET_DATE" | cut -d'-' -f2)
    DATE_15TH_NODASH="${YEAR}${MONTH}15"
    SIC_DIR_PART="_sic"
    SEARCH_DIR="${OUTPUT_PATH}/${DATASET_VERSION}/${VARIABLE}"
    PREFIX="run_${NUM_DAYS}days${SIC_DIR_PART}_${DATE_15TH_NODASH}_v"
    SMOOTHED_PARAMS_DIR=""
    # Only look under {OUTPUT_PATH}/{DATASET_VERSION}/{VARIABLE}/ so we never load another variable's params
    if [ -d "$SEARCH_DIR" ]; then
        MATCHING_DIRS=$(find "$SEARCH_DIR" -maxdepth 1 -type d -name "${PREFIX}*" 2>/dev/null | sort -V | tail -1)
        if [ -n "$MATCHING_DIRS" ]; then
            SMOOTHED_PARAMS_DIR="$MATCHING_DIRS"
            echo "INFO: Found smoothed params directory: $(basename "$SMOOTHED_PARAMS_DIR")"
        fi
    fi
    if [ -z "$SMOOTHED_PARAMS_DIR" ] || [ ! -d "$SMOOTHED_PARAMS_DIR" ]; then
        echo "WARNING: Could not find smoothed params directory for 15th of month (SIC run)"
        echo "  Searched for: ${SEARCH_DIR}/${PREFIX}*"
        SMOOTHED_PARAMS=""
    else
        H5_FILENAME="${H5_FILENAME:-IS2_interp_test_petty_v1.h5}"
        H5_FILE="${SMOOTHED_PARAMS_DIR}/${H5_FILENAME}"
        if [ -f "$H5_FILE" ]; then
            SMOOTHED_PARAMS="$H5_FILE"
            echo "INFO: Using smoothed params from: ${SMOOTHED_PARAMS}"
        else
            SMOOTHED_PARAMS=""
        fi
    fi
fi

echo "=========================================="
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Processing date: ${TARGET_DATE}"
echo "Total dates: ${#DATES[@]}"
echo "Num days: ${NUM_DAYS}"
echo "SIC (ice-edge guide): true (no SMAP)"
echo "Dataset version: ${DATASET_VERSION}"
echo "Variable: ${VARIABLE}"
echo "IS2 pref: ${IS2_PREF}"
echo "Smoothed params: ${SMOOTHED_PARAMS}"
echo "Output path: ${OUTPUT_PATH}"
echo "=========================================="

# Python script: use absolute path (edit if your repo lives elsewhere)
PYTHON_SCRIPT="${PYTHON_SCRIPT:-/explore/nobackup/people/aapetty/GitHub/GPSat/IS2_SMAP_GPSat_train.py}"

CMD=(python "${PYTHON_SCRIPT}" \
  --num_days "${NUM_DAYS}" \
  --smap "false" \
  --sic "true" \
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
