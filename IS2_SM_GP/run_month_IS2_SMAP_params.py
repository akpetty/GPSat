#!/usr/bin/env python3
"""
Simplified month runner: calls the single-day wrapper for every day of a month.
No parameter reuse (root script currently re-optimises per day).
"""
import argparse, calendar, datetime as dt, os, subprocess, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SINGLE_DAY_SCRIPT = os.path.join(SCRIPT_DIR, 'run_IS2_SMAP_GPSat_train.py')

def list_month_days(year: int, month: int):
    ndays = calendar.monthrange(year, month)[1]
    return [dt.date(year, month, d) for d in range(1, ndays+1)]

def main():
    p = argparse.ArgumentParser(description='Run root SMAP script for each day of month.')
    p.add_argument('--month', required=True, help='YYYY-MM')
    p.add_argument('--num_days', required=True, type=int, help='Days before/after target date')
    p.add_argument('--smap', required=True, type=str, help='Include SMAP data true/false')
    p.add_argument('--start_day', type=int, default=None, help='Optional start day override')
    p.add_argument('--end_day', type=int, default=None, help='Optional end day override')
    p.add_argument('--smoothed_params_file_path', type=str, default=None, help='Path to H5 file with smoothed params (prediction-only mode)')
    p.add_argument('--output_results_path', type=str, default=None, help='Full path to base results directory')
    p.add_argument('--dataset_version', type=str, default=None, help='Dataset version string (e.g., dev_v1, v1.0)')
    p.add_argument('--variable', type=str, default='thickness', help='Variable name (default: thickness)')
    args = p.parse_args()

    year, month = map(int, args.month.split('-'))
    all_days = list_month_days(year, month)
    if args.start_day or args.end_day:
        sd = args.start_day or 1
        ed = args.end_day or all_days[-1].day
        all_days = [dt.date(year, month, d) for d in range(sd, ed+1)]

    for d in all_days:
        d_str = d.strftime('%Y-%m-%d')
        print(f'Running day {d_str}')
        cmd = [sys.executable, SINGLE_DAY_SCRIPT,
               '--date', d_str,
               '--num_days', str(args.num_days),
               '--smap', args.smap]
        if args.smoothed_params_file_path:
            cmd.extend(['--smoothed_params_file_path', args.smoothed_params_file_path])
        if args.output_results_path:
            cmd.extend(['--output_results_path', args.output_results_path])
        if args.dataset_version:
            cmd.extend(['--dataset_version', args.dataset_version])
        if args.variable:
            cmd.extend(['--variable', args.variable])
        subprocess.check_call(cmd)

    print('Month run complete.')

if __name__ == '__main__':
    main()
