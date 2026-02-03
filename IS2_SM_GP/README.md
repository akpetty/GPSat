# IS2_SM_GP Production Workflow

## Notes

I've set this to False: check_config_compatible=False

## Environment stuff

export PATH="/home/aapetty/.conda/envs/is2_39_gp/bin:$PATH"
module load gcc/12.1.0
source activate is2_39_gp

python IS2_SMAP_GPSat_train.py --num_days 30 --smap true --target_date 2019-04-15 --output_results_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/ --dataset_version dev_v1 --variable thickness --smoothed_params_file_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/dev_v1/thickness/run_30days_smap_20190415_v01/IS2_interp_test_petty_v1.h5

Simplified production wrappers for running the root `IS2_SMAP_GPSat_train.py` (IS2 + optional SMAP).

## Components
- `IS2_SMAP_GPSat_train.py`: Main script for running IS2 + SMAP GPSat training/prediction.
- `run_IS2_SMAP_GPSat_train.py`: Thin wrapper that forwards arguments to root script.
- `run_month_IS2_SMAP_params.py`: Calls the wrapper for each day in a month (sequential, older approach - not recommended).
- `run_days_in_month_IS2_SMAP.sbatch`: SLURM array job for processing all days in a month (except the 15th) in parallel. **Recommended for processing full months.**
- `run_monthly_mid_daterange_IS2_SMAP.sbatch`: SLURM array job for processing the 15th (mid-month) of various months across a date range. **Recommended for mid-month runs across multiple months/years.**

## Single Day Usage

### Basic single day run:
```bash
python IS2_SM_GP/run_IS2_SMAP_GPSat_train.py \
  --date 2019-04-15 \
  --num_days 30 \
  --smap true \
  --output_results_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/ \
  --dataset_version dev_v1 \
  --variable thickness
```

### With smoothed params (prediction-only):
```bash
python IS2_SM_GP/run_IS2_SMAP_GPSat_train.py \
  --date 2019-04-15 \
  --num_days 30 \
  --smap true \
  --output_results_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/ \
  --dataset_version dev_v1 \
  --variable thickness \
  --smoothed_params_file_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/dev_v1/thickness/run_30days_smap_20190415_v01/IS2_interp_test_petty_v1.h5
```

**Notes:**
- `--date` becomes both the target date and results date; use `--target_date` explicitly if different.
- Directory names always use "run" as the prefix (hardcoded).
- `--dataset_version` is recommended to organize outputs by version.
- `--variable` defaults to `thickness` if not specified.

## Month Run (re-optimises each day) - Older Sequential Approach

**Note:** This is the older sequential approach. For parallel processing, use the SLURM array scripts instead.

```bash
python IS2_SM_GP/run_month_IS2_SMAP_params.py \
  --month 2019-04 \
  --num_days 30 \
  --smap true \
  --output_results_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/ \
  --dataset_version dev_v1 \
  --variable thickness
```

Optional subset of days:
```bash
python IS2_SM_GP/run_month_IS2_SMAP_params.py \
  --month 2019-04 \
  --num_days 30 \
  --smap true \
  --start_day 10 \
  --end_day 15 \
  --output_results_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/ \
  --dataset_version dev_v1 \
  --variable thickness
```

Month prediction-only using existing smoothed params (same file reused each day):
```bash
python IS2_SM_GP/run_month_IS2_SMAP_params.py \
  --month 2019-04 \
  --num_days 30 \
  --smap true \
  --load_smoothed_params \
  --smoothed_params_file_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/dev_v1/thickness/run_30days_smap_20190415_v01/IS2_interp_test_petty_v1.h5 \
  --output_results_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/ \
  --dataset_version dev_v1 \
  --variable thickness
```
Note: Smoothed hyperparameters are typically day-specific; reusing across days assumes spatial stationarity over the month—validate scientifically.

## SLURM Submission

All SLURM output files (`.out` and `.err`) are automatically saved to the `slurm_output/` subdirectory.

### Process all days in a month (except the 15th) - Parallel Array Job

**Script:** `run_days_in_month_IS2_SMAP.sbatch`

Processes all days of a given month in parallel, skipping the 15th (which is typically used for generating smoothed parameters).

**Usage:**
```bash
sbatch IS2_SM_GP/run_days_in_month_IS2_SMAP.sbatch YYYY-MM NUM_DAYS SMAP_FLAG DATASET_VERSION [VARIABLE] [SMOOTHED_PARAMS_FILE]
```

**Parameters:**
- `YYYY-MM`: Target month (e.g., `2023-02`)
- `NUM_DAYS`: Days before/after target date window
- `SMAP_FLAG`: `true` or `false` to include SMAP data
- `DATASET_VERSION`: Version identifier (e.g., `dev_v1`, `v1.0`) - **required**
- `VARIABLE`: Variable name (default: `thickness`) - **optional**
- `SMOOTHED_PARAMS_FILE`: Path to smoothed params H5 file - **optional** (defaults to 15th of same month)

**Examples:**
```bash
# Development run
sbatch IS2_SM_GP/run_days_in_month_IS2_SMAP.sbatch 2023-02 30 true dev_v1

# With explicit variable
sbatch IS2_SM_GP/run_days_in_month_IS2_SMAP.sbatch 2023-02 30 true dev_v1 thickness

# With custom smoothed params file
sbatch IS2_SM_GP/run_days_in_month_IS2_SMAP.sbatch 2023-02 30 true dev_v1 thickness /path/to/smoothed_params.h5
```

**Note:** If `SMOOTHED_PARAMS_FILE` is omitted, the script automatically looks for smoothed params from the 15th of the same month in the same version.

### Process mid-month dates (15th) across a date range - Parallel Array Job

**Script:** `run_monthly_mid_daterange_IS2_SMAP.sbatch`

Processes the 15th (mid-month) of various months across multiple years. Useful for generating smoothed parameters or processing specific dates.

**Usage:**
1. Edit the `DATES` array in the script to specify which dates to process
2. Run the script:
```bash
sbatch IS2_SM_GP/run_monthly_mid_daterange_IS2_SMAP.sbatch NUM_DAYS SMAP DATASET_VERSION [VARIABLE] [SMOOTHED_PARAMS_FILE]
```

**Parameters:**
- `NUM_DAYS`: Days before/after target date window
- `SMAP`: `true` or `false` to include SMAP data
- `DATASET_VERSION`: Version identifier (e.g., `dev_v1`, `v1.0`) - **required**
- `VARIABLE`: Variable name (default: `thickness`) - **optional**
- `SMOOTHED_PARAMS_FILE`: Path to smoothed params H5 file - **optional**

**Examples:**
```bash
# Process all dates in the DATES array
sbatch IS2_SM_GP/run_monthly_mid_daterange_IS2_SMAP.sbatch 30 true dev_v1

# With explicit variable
sbatch IS2_SM_GP/run_monthly_mid_daterange_IS2_SMAP.sbatch 30 true dev_v1 thickness
```

**Note:** The script automatically constructs smoothed params paths based on the 15th of the same month for each date in the array.

## Version Naming Strategy

The `--dataset_version` parameter organizes your runs and tracks global settings. Use the following naming convention:

### Format: `{stage}_{number}` or `{stage}_{description}`

**Stages:**
- **`dev_*`** - Development/experimental runs
  - Examples: `dev_v1`, `dev_v2`, `dev_test_params`
  - Use for testing different parameters, debugging, quick experiments

- **`test_*`** - Testing/validation runs
  - Examples: `test_v1`, `test_final_check`
  - Use for runs that are close to production but need validation

- **`v*`** or **`prod_*`** - Production runs
  - Examples: `dev_v1`, `v1.1`, `prod_v1`
  - Use for final, validated runs

### Version Config File

Each version automatically gets a `version_config.csv` file in the version directory (e.g., `/output/dev_v1/version_config.csv`) that tracks:
- Global settings (num_days, smap_enabled, etc.)
- Creation and last update timestamps
- Base output path

This makes it easy to see what settings were used for all runs in a given version.

### Example Structure:
```
output/
  dev_v1/              # First experimental version
    version_config.csv
    thickness/
      multiparams_30days_smap_20230201_v01/
      ...
  
  dev_v1/                # First production version
    version_config.csv
    thickness/
      ...
```

## Output Structure

Outputs are organized as:
```
{output_path}/
  {dataset_version}/
    version_config.csv          # Global settings for this version
    {variable}/
      run_{num_days}days_{smap}_{YYYYMMDD}_v{NN}/
        config.csv              # Run-specific config
        IS2_interp_test_petty_v1.h5
        *.png                   # Plots
        ...
```

**Example:**
```
/output/
  dev_v1/
    version_config.csv
    thickness/
      run_30days_smap_20230201_v01/
      run_30days_smap_20230202_v01/
      ...
```

## Notes
- **Version config:** Each version automatically gets a `version_config.csv` file tracking global settings (see Version Naming Strategy section).
- **Smoothed params:** The scripts automatically find smoothed params from the 15th of the same month if not explicitly provided.
- **SLURM outputs:** All `.out` and `.err` files are saved to `slurm_output/` subdirectory.
- **Parallel processing:** The array-based scripts process jobs in parallel and are recommended for production runs.
- **Environment:** Ensure correct conda env activation (`is2_39_gp` in sbatch scripts).

## Direct Root Script Run

You can bypass the wrapper and call the main script directly:
```bash
python IS2_SMAP_GPSat_train.py \
  --num_days 30 \
  --smap true \
  --target_date 2019-04-15 \
  --output_results_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/ \
  --dataset_version dev_v1 \
  --variable thickness
```

**Required parameters:**
- `--num_days`: Days before/after target date window
- `--smap`: `true` or `false` to include SMAP data
- `--target_date`: Target date in YYYY-MM-DD format

**Optional but recommended:**
- `--dataset_version`: Version identifier (e.g., `dev_v1`, `v1.0`)
- `--variable`: Variable name (default: `thickness`)
- `--output_results_path`: Base output directory path
- `--smoothed_params_file_path`: Path to H5 file with smoothed parameters for loading

## Loading Smoothed Parameters (Reusing Hyperparameters)
To skip optimisation and reuse previously smoothed hyperparameters, supply either:
- `--load_smoothed_params` (auto-derive H5 path in new results directory; expects you manually place/copy file), or
- `--smoothed_params_file_path /path/to/previous_run/IS2_interp_test_petty_v1.h5` (implies load flag).

When `--load_smoothed_params` is active and the H5 file lives in a different source directory, the script automatically copies into the new results directory:
- The H5 results file (with smoothed tables)
- Any `*SMOOTHED*.json` smoothing config files
- The source `config.csv` (renamed to `config_source.csv` to preserve this run's own `config.csv`)

Example (direct):
```bash
python IS2_SMAP_GPSat_train.py \
  --num_days 30 \
  --smap true \
  --target_date 2019-04-15 \
  --output_results_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/ \
  --dataset_version dev_v1 \
  --variable thickness \
  --smoothed_params_file_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/dev_v1/thickness/run_30days_smap_20190415_v01/IS2_interp_test_petty_v1.h5
```

Example (wrapper):
```bash
python IS2_SM_GP/run_IS2_SMAP_GPSat_train.py \
  --date 2019-04-15 \
  --num_days 30 \
  --smap true \
  --output_results_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/ \
  --dataset_version dev_v1 \
  --variable thickness \
  --smoothed_params_file_path /explore/nobackup/people/aapetty/GitHub/GPSat/output/dev_v1/thickness/run_30days_smap_20190415_v01/IS2_interp_test_petty_v1.h5
```

Add `--output_results_path /full/path` to place new versioned output directories elsewhere (e.g., nobackup). If omitted, defaults to `./results` relative to repo root.

Outputs will be versioned as usual; loaded runs prepend `load_sm_params_` internally to the base descriptor for clarity.
