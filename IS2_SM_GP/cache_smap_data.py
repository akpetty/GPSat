#!/usr/bin/env python3
"""
Script to cache SMAP data and generate availability report.

This script downloads and caches SMAP sea ice thickness data from the University of Bremen
for a specified date range, and generates a CSV file reporting which dates have data available.

Usage:
    python cache_smap_data.py --start_date 2023-01-01 --end_date 2023-12-31
"""

import os
import sys
import argparse
import pandas as pd
import xarray as xr
import fsspec
from datetime import datetime, timedelta

def check_and_cache_smap_date(date_str, cache_dir):
    """
    Check if SMAP data exists for a date and cache it if available.
    
    Parameters:
    -----------
    date_str : str
        Date string in format 'YYYY-MM-DD'
    cache_dir : str
        Directory to cache SMAP files
        
    Returns:
    --------
    result : dict
        Dictionary with 'success', 'cached', 'missing' status
    """
    # Convert date format from 'YYYY-MM-DD' to 'YYYYMMDD'
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    date_compact = date_obj.strftime('%Y%m%d')
    
    # SMAP data URL
    url = f'https://data.seaice.uni-bremen.de/smos_smap/netCDF/north/{date_compact[0:4]}/{date_compact}_north_mix_sit_v300.nc'
    
    # Set up local cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Local cache file path
    cache_filename = f"{date_compact}_north_mix_sit_v300.nc"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    result = {
        'date': date_str,
        'success': False,
        'cached': False,
        'missing': False
    }
    
    # Check if file exists in cache
    if os.path.exists(cache_path):
        result['success'] = True
        result['cached'] = True
        return result
    
    # Try to download
    try:
        fs = fsspec.open(url)
        with fs.open() as remote_file:
            with open(cache_path, 'wb') as local_file:
                local_file.write(remote_file.read())
        result['success'] = True
        result['cached'] = False
        return result
    except Exception as e:
        # File doesn't exist on server
        result['missing'] = True
        return result


def cache_smap_date_range(start_date, end_date, cache_dir, skip_existing=True, verbose=True):
    """
    Cache SMAP data for a date range and generate availability report.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    cache_dir : str
        Directory to cache SMAP files
    skip_existing : bool
        If True, skip dates that are already cached
    verbose : bool
        If True, print progress messages
        
    Returns:
    --------
    summary_df : pandas.DataFrame
        DataFrame with availability report for each date
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    summary_records = []
    current_dt = start_dt
    
    total_days = (end_dt - start_dt).days + 1
    if verbose:
        print(f"Processing {total_days} days from {start_date} to {end_date}")
        print(f"Cache directory: {cache_dir}\n")
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y-%m-%d')
        
        if verbose:
            print(f"[{date_str}] ", end='', flush=True)
        
        # Check if already cached and skip_existing is True
        if skip_existing:
            date_compact = current_dt.strftime('%Y%m%d')
            cache_filename = f"{date_compact}_north_mix_sit_v300.nc"
            cache_path = os.path.join(cache_dir, cache_filename)
            if os.path.exists(cache_path):
                if verbose:
                    print("Already cached")
                summary_records.append({
                    'date': date_str,
                    'success': True,
                    'cached': True,
                    'missing': False
                })
                current_dt += timedelta(days=1)
                continue
        
        # Try to download
        result = check_and_cache_smap_date(date_str, cache_dir)
        summary_records.append(result)
        
        if verbose:
            if result['success']:
                if result['cached']:
                    print("Already cached")
                else:
                    print("Downloaded")
            else:
                print("Missing")
        
        current_dt += timedelta(days=1)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_records)
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Cache SMAP data and generate availability report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cache data for a year
  python cache_smap_data.py --start_date 2023-01-01 --end_date 2023-12-31
  
  # Cache data with custom cache directory
  python cache_smap_data.py --start_date 2023-01-01 --end_date 2023-12-31 --cache_dir /path/to/cache
        """
    )
    
    parser.add_argument('--start_date', type=str, required=True,
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True,
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--cache_dir', type=str, 
                       default='/explore/nobackup/people/aapetty/SMAP/thickness_cache/',
                       help='Directory to cache SMAP files')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Output CSV file path (default: smap_availability_YYYYMMDD-YYYYMMDD.csv)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='Skip dates that are already cached (default: True)')
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                       help='Re-download even if file exists in cache')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        if start_dt > end_dt:
            print("Error: start_date must be before end_date")
            sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD format. {e}")
        sys.exit(1)
    
    # Generate summary
    summary_df = cache_smap_date_range(
        args.start_date, 
        args.end_date, 
        args.cache_dir,
        args.skip_existing,
        verbose=not args.quiet
    )
    
    # Determine output CSV path
    if args.output_csv is None:
        start_compact = start_dt.strftime('%Y%m%d')
        end_compact = end_dt.strftime('%Y%m%d')
        args.output_csv = f'smap_availability_{start_compact}-{end_compact}.csv'
    
    # Save summary to CSV
    summary_df.to_csv(args.output_csv, index=False)
    print(f"\nSummary saved to: {args.output_csv}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    total_days = len(summary_df)
    successful = summary_df['success'].sum()
    missing = summary_df['missing'].sum()
    cached = summary_df['cached'].sum()
    downloaded = successful - cached
    
    print(f"Total days: {total_days}")
    print(f"Available (cached or downloaded): {successful} ({100*successful/total_days:.1f}%)")
    print(f"  - Already cached: {cached}")
    print(f"  - Downloaded: {downloaded}")
    print(f"Missing: {missing} ({100*missing/total_days:.1f}%)")


if __name__ == '__main__':
    main()
