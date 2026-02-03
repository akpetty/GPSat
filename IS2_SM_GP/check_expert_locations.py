#!/usr/bin/env python3
"""
Script to check which expert locations were used in a GPSat run.

Usage:
    python check_expert_locations.py <h5_file_path>
    
Example:
    python check_expert_locations.py /path/to/IS2_interp_test_petty_v1.h5
"""

import sys
import pandas as pd
import numpy as np

def check_expert_locations(h5_path):
    """Check which expert locations were used in the run."""
    
    print(f"Reading H5 file: {h5_path}\n")
    
    with pd.HDFStore(h5_path, mode='r') as store:
        # List all available tables
        all_keys = [k.lstrip('/') for k in store.keys()]
        print(f"Available tables: {all_keys}\n")
        
        # Check for expert locations tables
        expert_tables = [k for k in all_keys if 'expert_locs' in k.lower()]
        run_details_tables = [k for k in all_keys if 'run_details' in k.lower()]
        pred_tables = [k for k in all_keys if 'preds' in k.lower()]
        
        print("=" * 70)
        print("EXPERT LOCATIONS SUMMARY")
        print("=" * 70)
        
        # Read expert locations
        if expert_tables:
            for table in expert_tables:
                print(f"\nTable: {table}")
                try:
                    expert_locs = store.select(table)
                    print(f"  Total expert locations: {len(expert_locs)}")
                    if len(expert_locs) > 0:
                        print(f"  Columns: {list(expert_locs.columns)}")
                        if 'x' in expert_locs.columns and 'y' in expert_locs.columns:
                            print(f"  X range: {expert_locs['x'].min():.2f} to {expert_locs['x'].max():.2f}")
                            print(f"  Y range: {expert_locs['y'].min():.2f} to {expert_locs['y'].max():.2f}")
                        if 'time' in expert_locs.columns:
                            times = expert_locs['time'].unique()
                            print(f"  Time values: {times}")
                except Exception as e:
                    print(f"  Error reading table: {e}")
        else:
            print("\nNo expert_locs tables found")
        
        # Read run details to see which experts were actually processed
        print("\n" + "=" * 70)
        print("RUN DETAILS (Which experts were processed)")
        print("=" * 70)
        
        if run_details_tables:
            for table in run_details_tables:
                print(f"\nTable: {table}")
                try:
                    run_details = store.select(table)
                    print(f"  Total runs: {len(run_details)}")
                    if len(run_details) > 0:
                        print(f"  Columns: {list(run_details.columns)}")
                        
                        # Count successful vs failed runs
                        if 'optimise_success' in run_details.columns:
                            success = run_details['optimise_success'].sum()
                            failed = len(run_details) - success
                            print(f"  Successful optimizations: {success}")
                            print(f"  Failed optimizations: {failed}")
                        
                        if 'num_obs' in run_details.columns:
                            print(f"  Observations per expert:")
                            print(f"    Min: {run_details['num_obs'].min()}")
                            print(f"    Max: {run_details['num_obs'].max()}")
                            print(f"    Mean: {run_details['num_obs'].mean():.1f}")
                            print(f"    Median: {run_details['num_obs'].median():.1f}")
                        
                        # Show coordinate columns if available
                        coord_cols = [c for c in run_details.columns if c in ['x', 'y', 'time']]
                        if coord_cols:
                            print(f"  Coordinate columns: {coord_cols}")
                            for col in coord_cols:
                                if col in run_details.columns:
                                    vals = run_details[col].unique()
                                    if len(vals) <= 10:
                                        print(f"    {col}: {vals}")
                                    else:
                                        print(f"    {col}: {len(vals)} unique values, range: {vals.min()} to {vals.max()}")
                except Exception as e:
                    print(f"  Error reading table: {e}")
        else:
            print("\nNo run_details tables found")
        
        # Check predictions to see which experts made predictions
        print("\n" + "=" * 70)
        print("PREDICTIONS (Which experts made predictions)")
        print("=" * 70)
        
        if pred_tables:
            for table in pred_tables:
                print(f"\nTable: {table}")
                try:
                    preds = store.select(table)
                    print(f"  Total predictions: {len(preds)}")
                    if len(preds) > 0:
                        print(f"  Columns: {list(preds.columns)}")
                        
                        # Count unique expert locations that made predictions
                        coord_cols = [c for c in preds.columns if c in ['x', 'y', 'time']]
                        if coord_cols:
                            unique_experts = preds[coord_cols].drop_duplicates()
                            print(f"  Unique expert locations with predictions: {len(unique_experts)}")
                            
                            # Check for NaN predictions (failed experts)
                            if 'f*' in preds.columns:
                                nan_preds = preds['f*'].isna().sum()
                                valid_preds = len(preds) - nan_preds
                                print(f"  Valid predictions: {valid_preds}")
                                print(f"  NaN predictions (failed experts): {nan_preds}")
                except Exception as e:
                    print(f"  Error reading table: {e}")
        else:
            print("\nNo prediction tables found")
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        # Compare expert locations vs actually processed
        if expert_tables and run_details_tables:
            try:
                expert_locs = store.select(expert_tables[0])
                run_details = store.select(run_details_tables[0])
                
                # Get coordinate columns
                expert_coords = [c for c in expert_locs.columns if c in ['x', 'y', 'time']]
                run_coords = [c for c in run_details.columns if c in ['x', 'y', 'time']]
                
                if expert_coords and run_coords and set(expert_coords) == set(run_coords):
                    expert_set = set(tuple(row) for row in expert_locs[expert_coords].values)
                    run_set = set(tuple(row) for row in run_details[run_coords].values)
                    
                    processed = len(run_set)
                    total = len(expert_set)
                    skipped = total - processed
                    
                    print(f"\nTotal expert locations: {total}")
                    print(f"Expert locations processed: {processed}")
                    print(f"Expert locations skipped: {skipped}")
                    if total > 0:
                        print(f"Processing rate: {100*processed/total:.1f}%")
                else:
                    print("\nCannot compare - coordinate columns don't match")
            except Exception as e:
                print(f"\nError comparing tables: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_expert_locations.py <h5_file_path>")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    check_expert_locations(h5_path)




