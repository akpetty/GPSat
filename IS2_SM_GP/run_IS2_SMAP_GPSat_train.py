#!/usr/bin/env python3
"""
Thin wrapper that delegates single-day runs to the root `IS2_SMAP_GPSat_train.py` script.
Keeps a consistent interface for production tooling inside `IS2_SM_GP`.
"""
import os, sys, argparse, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET_SCRIPT = os.path.join(ROOT, 'IS2_SMAP_GPSat_train.py')

def str2bool(s):
    return s.lower() in ['true','1','yes','y']

def main():
    parser = argparse.ArgumentParser(description="Wrapper: run root IS2_SMAP_GPSat_train.py")
    parser.add_argument("--date", required=True, help="Target date YYYY-MM-DD (used also as results date)")
    parser.add_argument("--target_date", required=False, help="Alias for --date if provided")
    parser.add_argument("--num_days", required=True, type=int, help="Days before/after target date")
    parser.add_argument("--smap", required=True, type=str, help="Include SMAP data true/false")
    parser.add_argument("--smoothed_params_file_path", type=str, default=None, help="Path to H5 file containing smoothed params (prediction-only mode)")
    parser.add_argument("--output_results_path", type=str, default=None, help="Full path to base results directory (default: ./results)")
    parser.add_argument("--dataset_version", type=str, default=None, help="Dataset version string (e.g., dev_v1, v1.0)")
    parser.add_argument("--variable", type=str, default="thickness", help="Variable name (default: thickness)")
    parser.add_argument("--smap_use_prediction_day_only", action='store_true', help="Only load SMAP data for the prediction day (faster, but less SMAP data).")
    args = parser.parse_args()

    target = args.target_date or args.date

    cmd = [sys.executable, TARGET_SCRIPT,
           '--num_days', str(args.num_days),
           '--smap', args.smap,
           '--target_date', target,
           '--date', args.date]

    if args.smoothed_params_file_path:
        cmd.extend(['--smoothed_params_file_path', args.smoothed_params_file_path])
    if args.output_results_path:
        cmd.extend(['--output_results_path', args.output_results_path])
    if args.dataset_version:
        cmd.extend(['--dataset_version', args.dataset_version])
    if args.variable:
        cmd.extend(['--variable', args.variable])
    if args.smap_use_prediction_day_only:
        cmd.append('--smap_use_prediction_day_only')

    print('Executing root script:', ' '.join(cmd))
    subprocess.check_call(cmd)

if __name__ == '__main__':
    main()
