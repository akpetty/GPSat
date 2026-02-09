import os
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = info, 2 = warning, 3 = error
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage since CUDA runtime libraries are missing

# Fix TensorFlow/Keras compatibility issues
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Add memory management
import gc
import psutil
import sys

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()
    print_memory_usage()

import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
#import regionmask
import re
from global_land_mask import globe
from datetime import datetime, timedelta
import s3fs
from io import StringIO  
import intake
import itertools
import glob
import random
from functools import reduce
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import sys, os
from dask.diagnostics import ProgressBar
#import geopandas as gp
import logging
import time  # Add time module for timing
import traceback  # Move traceback import to top level
import argparse
import resource  # Add resource module for memory tracking
import fsspec  # Add fsspec for SMAP data loading
import shutil  # For copying smoothed parameter artifacts


import re

# Suppress HDF5 warnings
import h5py
h5py._errors.silence_errors()
# Import local modules
from extra_funcs import read_IS2SITMOGR4, along_track_preprocess, bin_to_IS2, cdr_preprocess_nh, load_sic_data_for_date

# Configure TensorFlow to handle GPU/CPU gracefully
import tensorflow as tf

# Fix Keras compatibility for older TensorFlow versions
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
try:
    import keras
    # Force use of legacy Keras if needed
    if not hasattr(keras, '__internal__'):
        import tensorflow.keras as keras
except ImportError:
    pass

# Additional compatibility for older TensorFlow versions
import tensorflow as tf
tf_version = tf.__version__
print(f"TensorFlow version: {tf_version}")

# Set compatibility flags for older versions
if tf_version.startswith('2.14') or tf_version.startswith('2.15'):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print("Using TensorFlow 2.14/2.15 compatibility settings")

# Ensure griddata is properly imported
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU devices found. Running on CPU.")
    print("Available devices:", tf.config.list_physical_devices())
else:
    print(f"Found {len(physical_devices)} GPU devices. Using GPU.")
    print("GPU devices:", physical_devices)
    
    # Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except RuntimeError as e:
            print(f"Error setting memory growth for {device}: {e}")
    
    # Optional: Set memory limit (uncomment if needed)
    # tf.config.set_logical_device_configuration(
    #     physical_devices[0],
    #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    # )


import GPSat
from GPSat import get_data_path, get_parent_path
from GPSat.dataprepper import DataPrep
from GPSat.dataloader import DataLoader
from GPSat.utils import stats_on_vals, WGS84toEASE2, EASE2toWGS84, cprint, grid_2d_flatten, get_weighted_values, dataframe_to_2d_array
from GPSat.local_experts import LocalExpertOI, get_results_from_h5file
from GPSat.plot_utils import plot_wrapper, plot_pcolormesh, get_projection, plot_pcolormesh_from_results_data, plot_hyper_parameters
from GPSat.postprocessing import smooth_hyperparameters

print('loaded envs')


# Reduce noisy logs but keep warnings/errors visible
for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.WARNING)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

def load_smap_data_for_date(date_str, IS2, config=None):
    """
    Load SMAP data for a specific date
    
    Parameters:
    -----------
    date_str : str
        Date string in format 'YYYY-MM-DD'
    IS2 : xarray.Dataset
        IS2 dataset for coordinate reference
    config : dict, optional
        Configuration dictionary containing SMAP parameters
        
    Returns:
    --------
    smap_data : pandas.DataFrame
        DataFrame containing SMAP thickness data
    smap_data_gridded : xarray.Dataset
        Gridded SMAP data binned to IS2 grid
    """
    # Convert date format from 'YYYY-MM-DD' to 'YYYYMMDD'
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    date_compact = date_obj.strftime('%Y%m%d')
    
    print(f"Loading SMAP data for date: {date_str} ({date_compact})")
    
    # SMAP data URL
    url = f'https://data.seaice.uni-bremen.de/smos_smap/netCDF/north/{date_compact[0:4]}/{date_compact}_north_mix_sit_v300.nc'
    
    # Set up local cache directory
    cache_dir = config.get('smap_cache_dir', os.path.join(os.path.expanduser('~'), '.cache', 'smap_data')) if config else os.path.join(os.path.expanduser('~'), '.cache', 'smap_data')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Local cache file path
    cache_filename = f"{date_compact}_north_mix_sit_v300.nc"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    try:
        # Check if file exists in cache
        if os.path.exists(cache_path):
            print(f"  Using cached SMAP file: {cache_path}")
            ds_smap = xr.open_dataset(cache_path)
        else:
            print(f"  Downloading SMAP data from: {url}")
            print(f"  Caching to: {cache_path}")
            # Download and cache the file using fsspec
            try:
                fs = fsspec.open(url)
                with fs.open() as remote_file:
                    with open(cache_path, 'wb') as local_file:
                        local_file.write(remote_file.read())
                print(f"  Download complete, loading from cache")
                ds_smap = xr.open_dataset(cache_path)
            except Exception as download_error:
                # If download fails, try direct access (no cache)
                print(f"  Warning: Cache download failed ({download_error}), trying direct access")
                if os.path.exists(cache_path):
                    os.remove(cache_path)  # Remove partial file
                fs = fsspec.open(url)
                ds_smap = xr.open_dataset(fs.open())
        
        # Load lat/lon coordinates
        data_path_ll = '/explore/nobackup/people/aapetty/Data/Other/NSIDC0771_LatLon_PS_N12.5km_v1.0.nc'
        ds_lonlat = xr.open_dataset(data_path_ll)
        
        # Add lat/lon coordinates to SMAP dataset
        ds_smap = ds_smap.assign_coords(
            latitude=(('y', 'x'), ds_lonlat['latitude'].values[::-1, :]),
            longitude=(('y', 'x'), ds_lonlat['longitude'].values[::-1, :])
        )
        
        # Convert thickness from cm to meters
        for var in ['smos_thickness', 'smap_thickness', 'combined_thickness']:
            if var in ds_smap:
                ds_smap[var] = ds_smap[var] / 100.0
        
        # Use combined thickness as the main variable
        thickness_var = 'combined_thickness'
        if thickness_var not in ds_smap:
            # Fallback to SMAP thickness if combined not available
            thickness_var = 'smap_thickness'
            if thickness_var not in ds_smap:
                thickness_var = 'smos_thickness'
        
        print(f"Using {thickness_var} for SMAP data")
        
        # Extract thickness data
        smap_thickness = ds_smap[thickness_var]
        
        # Apply quality filters using config parameters if available
        thickness_min = config.get('smap_thickness_min', 0.0) if config else 0.0
        thickness_max = config.get('smap_thickness_max', 0.5) if config else 0.5
        
        print(f"SMAP thickness filtering: min={thickness_min}m, max={thickness_max}m")
        print(f"SMAP thickness range before filtering: {smap_thickness.min().values:.3f}m to {smap_thickness.max().values:.3f}m")
        
        smap_thickness = smap_thickness.where((smap_thickness >= thickness_min) & (smap_thickness <= thickness_max))
        
        print(f"SMAP thickness range after filtering: {smap_thickness.min().values:.3f}m to {smap_thickness.max().values:.3f}m")
        print(f"SMAP valid points before filtering: {np.sum(~np.isnan(smap_thickness.values))}")
        print(f"SMAP valid points after filtering: {np.sum(~np.isnan(smap_thickness.where((smap_thickness >= thickness_min) & (smap_thickness <= thickness_max)).values))}")
        
        # Apply coarsening if specified
        coarsen_factor = config.get('smap_coarsen_factor', 1) if config else 1
        if coarsen_factor > 1:
            print(f'Coarsening SMAP data by factor of {coarsen_factor}')
            # Coarsen by taking every nth point in both dimensions
            smap_thickness = smap_thickness.isel(x=slice(None, None, coarsen_factor),
                                               y=slice(None, None, coarsen_factor))
            print(f"SMAP data shape after coarsening: {smap_thickness.shape}")
        
        # Apply region filtering if specified
        apply_region_filter = config.get('smap_apply_region_filter', False) if config else False
        if apply_region_filter and 'region_mask' in IS2.data_vars:
            print("Applying region filtering to SMAP data...")
            try:
                # Get region mask for the target date (or closest available)
                region_mask_data = IS2.region_mask.sel(time=date_str, method='nearest')
                print(f"Using region_mask for date: {date_str}")
                
                # Interpolate region_mask to SMAP grid
                from scipy.interpolate import griddata
                
                # Create meshgrid for IS2 coordinates
                x_mesh_is2, y_mesh_is2 = np.meshgrid(IS2.x.values, IS2.y.values)
                
                # Flatten the coordinates and values
                x_coords_is2 = x_mesh_is2.flatten()
                y_coords_is2 = y_mesh_is2.flatten()
                region_values_is2 = region_mask_data.values.flatten()
                
                # Create points for interpolation
                points_is2 = np.column_stack((x_coords_is2, y_coords_is2))
                
                # Create meshgrid for SMAP coordinates
                x_mesh_smap, y_mesh_smap = np.meshgrid(smap_thickness.x.values, smap_thickness.y.values)
                
                # Get SMAP location coordinates
                smap_points = np.column_stack((x_mesh_smap.flatten(), y_mesh_smap.flatten()))
                
                # Interpolate region values to SMAP grid
                smap_regions = griddata(points_is2, region_values_is2, smap_points, method='nearest')
                smap_regions = smap_regions.reshape(smap_thickness.shape)
                
                # Get allowed regions from config
                allowed_regions = config.get('allowed_regions', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                
                # Create region mask for SMAP data
                region_mask_smap = np.isin(smap_regions, allowed_regions)
                
                # Apply region mask to thickness data
                smap_thickness = smap_thickness.where(region_mask_smap)
                
                print(f"SMAP valid points before region filtering: {np.sum(~np.isnan(smap_thickness.values))}")
                print(f"SMAP regions found: {sorted(np.unique(smap_regions[~np.isnan(smap_regions)]))}")
                print(f"SMAP regions in allowed list: {sorted(set(np.unique(smap_regions[~np.isnan(smap_regions)])).intersection(set(allowed_regions)))}")
                
            except Exception as e:
                print(f"Error applying region filtering to SMAP data: {str(e)}")
                print("Continuing without region filtering...")
        
        # Exclude SMAP in specified regions (e.g. Central Arctic = 1) using IS2SITMOGR4 region mask
        smap_exclude_regions = config.get('smap_exclude_regions', []) if config else []
        if smap_exclude_regions and 'region_mask' in IS2.data_vars:
            print(f"Excluding SMAP data in regions {smap_exclude_regions} (e.g. Central Arctic)...")
            try:
                region_mask_data = IS2.region_mask.sel(time=date_str, method='nearest')
                x_mesh_is2, y_mesh_is2 = np.meshgrid(IS2.x.values, IS2.y.values)
                points_is2 = np.column_stack((x_mesh_is2.flatten(), y_mesh_is2.flatten()))
                region_values_is2 = region_mask_data.values.flatten()
                x_mesh_s, y_mesh_s = np.meshgrid(smap_thickness.x.values, smap_thickness.y.values)
                smap_points = np.column_stack((x_mesh_s.flatten(), y_mesh_s.flatten()))
                smap_regions = griddata(points_is2, region_values_is2, smap_points, method='nearest')
                smap_regions = smap_regions.reshape(smap_thickness.shape)
                exclude_mask = np.isin(smap_regions, smap_exclude_regions)
                n_excluded = int(np.sum(exclude_mask))
                smap_thickness = smap_thickness.where(~exclude_mask)
                print(f"  Excluded {n_excluded} SMAP points in regions {smap_exclude_regions} (IS2SITMOGR4 region mask)")
            except Exception as e:
                print(f"Error excluding SMAP regions: {e}")
                print("Continuing without region exclusion...")
        
        # Create meshgrid from coordinates
        x_mesh, y_mesh = np.meshgrid(smap_thickness.x.values, smap_thickness.y.values)
        
        # Get valid data points
        valid_mask = ~np.isnan(smap_thickness.values)
        valid_thickness = smap_thickness.values[valid_mask]
        
        # Create DataFrame with valid data
        smap_data = pd.DataFrame({
            'x': x_mesh[valid_mask].flatten(),
            'y': y_mesh[valid_mask].flatten(),
            'ice_thickness': valid_thickness.flatten(),
            'time': pd.to_datetime(date_str)
        })
        
        # Bin to IS2 grid
        try:
            smap_data_gridded = bin_to_IS2(smap_data, IS2)
        except Exception as e:
            print(f'Error in binning SMAP data: {str(e)}')
            print('Error details:', e.__class__.__name__)
            print('Traceback:', traceback.format_exc())
            raise
        
        print(f"SMAP data loaded successfully: {len(smap_data)} valid points")
        
        return smap_data, smap_data_gridded
        
    except Exception as e:
        print(f"Error loading SMAP data for {date_str}: {str(e)}")
        print("Returning empty data")
        # Return empty data structures
        empty_df = pd.DataFrame(columns=['x', 'y', 'ice_thickness', 'time'])
        empty_gridded = xr.Dataset()
        return empty_df, empty_gridded


def load_data_around_target_date(config=None, IS2=None, plus_smap=False, plus_sic=False):
    """
    Load data for N days around a target date
    """
    target_date = config['target_date']
    num_days_before_after = config['num_days_before_after']
    beam = config['beam']
    
    # Convert single beam to list for consistent processing
    if isinstance(beam, str):
        beams = [beam]
    else:
        beams = beam
    
    along_track_data_all = []
    smap_data_all = []
    smap_data_gridded_all = []
    along_track_data_gridded_all = []
    
    # Parse target date
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Calculate date range
    start_dt = target_dt - timedelta(days=num_days_before_after)
    end_dt = target_dt + timedelta(days=num_days_before_after)
    
    val_col = config.get('val_col', 'ice_thickness')
    def _preprocess(ds):
        return along_track_preprocess(ds, data_variable=val_col)
    
    # Initialize S3 filesystem
    s3 = s3fs.S3FileSystem(anon=True)
    
    # Process each day in range
    current_dt = start_dt
    while current_dt <= end_dt:
        year_str = str(current_dt.year)
        month_str = f"{current_dt.month:02d}"
        day_str = f"{current_dt.day:02d}"
        current_date_str = f"{year_str}-{month_str}-{day_str}"
        
        print(f"Date of data loading: {current_date_str}, beams: {beams}")
        
        # Load SMAP data for the current date (only when plus_smap)
        use_prediction_day_only = config.get('smap_use_prediction_day_only', False) if config else False
        should_load_smap = plus_smap
        if should_load_smap and use_prediction_day_only:
            should_load_smap = (current_date_str == target_date)
            if should_load_smap:
                print(f"Loading SMAP data only for prediction day: {current_date_str}")
            else:
                print(f"Skipping SMAP data for {current_date_str} (not prediction day)")
        
        if should_load_smap:
            try:
                smap_data, smap_data_gridded = load_smap_data_for_date(current_date_str, IS2, config)
                if len(smap_data) > 0:
                    smap_data_all.append(smap_data)
                    smap_data_gridded_all.append(smap_data_gridded)
            except Exception as e:
                print(f'Error loading SMAP data for {current_date_str}: {e}')
                continue
        
        # Load SIC (ice-edge guide: 0 where SIC < 0.15) when plus_sic and not using SMAP
        # By default load SIC only for the prediction target date (sic_use_prediction_day_only=True)
        sic_use_prediction_day_only = config.get('sic_use_prediction_day_only', True) if config else True
        should_load_sic = plus_sic and not plus_smap
        if should_load_sic and sic_use_prediction_day_only:
            should_load_sic = (current_date_str == target_date)
            if should_load_sic:
                print(f"Loading SIC only for prediction day: {current_date_str}")
            # else: skip SIC for this day (not prediction day)
        if should_load_sic:
            try:
                sic_data, sic_data_gridded = load_sic_data_for_date(current_date_str, IS2, config)
                if len(sic_data) > 0:
                    smap_data_all.append(sic_data)
                    smap_data_gridded_all.append(sic_data_gridded)
            except Exception as e:
                print(f'Error loading SIC for {current_date_str}: {e}')
        
        # Load IS2 along-track data for each beam
        for beam_name in beams:
            print(f"  Processing beam: {beam_name}")
            
            # Construct file pattern for IS2 data
            # Read IS2 along-track path and version from config (with sensible defaults)
            base_is2_path = config.get('is2_alongtrack_base_path')

            # Construct file pattern for IS2 data
            file_pattern = os.path.join(
                config.get('is2_alongtrack_base_path'), f"{year_str}-{month_str}/data/sm/",
                f'IS2SITDAT4_01_{year_str}{month_str}{day_str}*{beam_name}*sm.nc'
            )
            file_list = glob.glob(file_pattern)
            print(f"    Found {len(file_list)} IS-2 files for {current_date_str}, beam {beam_name}")
            
            # Load IS2 along-track data for this beam
            for file in file_list:
                print('    Reading file:', file)
                try:
                    along_track_data = xr.open_mfdataset(file, engine='h5netcdf', preprocess=_preprocess, combine='nested')
                    # Subsample based on N_subsample to reduce input data size
                    along_track_data = along_track_data.isel(along_track_distance_section=slice(None, None, config['N_subsample']))  
                    # Print number of rows loaded from this file
                    try:
                        n_rows = along_track_data.sizes['along_track_distance_section']
                        print(f"      Loaded {n_rows} rows from file: {file}")
                    except Exception as e:
                        print(f"      Could not determine number of rows for file: {file} (error: {e})")
                    along_track_data_all.append(along_track_data)
                    along_track_data_gridded_all.append(bin_to_IS2(along_track_data, IS2, val_col=val_col))
                    along_track_data.close()
                except KeyError as e:
                    print(f"Error processing file {file}: {e}")
                    continue
        
        current_dt += timedelta(days=1)
    
    # Concatenate the data
    if along_track_data_all:
        along_track_data_all = xr.concat(along_track_data_all, 'along_track_distance_section')
        along_track_data_gridded_all = xr.concat(along_track_data_gridded_all, 'time')
    else:
        print("Warning: No along-track data found")
        along_track_data_all = xr.Dataset()
        along_track_data_gridded_all = xr.Dataset()
    
    return along_track_data_all, along_track_data_gridded_all, smap_data_all, smap_data_gridded_all

def str2bool(s):
        return s.lower() in ['true', '1', 'yes', 'y']

def plot_thickness_comparison(dataset, target_date, figsize=(15, 5), extra_var='', out_str='', days_str='', save_dir='./results/', cmax=5, variable='thickness'):
    """
    Create a comparison plot of two thickness/freeboard/snow_depth datasets with their difference.
    For freeboard and snow_depth: main colorbar capped at 0.6 m, difference -0.2 to 0.2 m.
    """
    array1 = dataset.SIT.isel(time=0)
    array2 = dataset.SIT_input
    lon = dataset.lon
    lat = dataset.lat
    
    # Variable-dependent limits and labels
    if variable in ('freeboard', 'snow_depth'):
        cmax_main = 0.6
        diff_vmin, diff_vmax = -0.2, 0.2
        var_label = variable.replace('_', ' ').title()  # "Freeboard" or "Snow Depth"
    else:
        cmax_main = cmax
        diff_vmin, diff_vmax = -2.0, 2.0
        var_label = 'Sea ice thickness'
    
    # Create figure and subplots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # Create subplots with orthographic projection
    axs = [fig.add_subplot(gs[0, x], projection=ccrs.Orthographic(-45, 90)) for x in range(2)]
    ax_diff = fig.add_subplot(gs[0, 2], projection=ccrs.Orthographic(-45, 90))
    
    # Calculate difference
    diffs = array1 - array2
    
    # Plot settings
    extent = [-180, 180, 55, 90]
    thickness_cmap = 'viridis'
    diff_cmap = 'RdBu_r'
    
    # Plot main variable maps
    thickness_plots = []
    for ax, data, title in zip(axs, [array1, array2], 
                              [f'(a) GPSat {var_label} {extra_var} ({target_date})',
                               f'(b) Binned input data +/- '+days_str+' days']):
        p = ax.pcolormesh(lon, lat, data,
                         vmin=0., vmax=cmax_main,
                         transform=ccrs.PlateCarree(),
                         cmap=thickness_cmap,
                         rasterized=True)
        thickness_plots.append(p)
        
        ax.coastlines(resolution='50m', linewidth=0.15, color='black', zorder=10)
        ax.add_feature(cfeature.LAND, color='0.95', zorder=0)
        ax.add_feature(cfeature.LAKES, color='grey', zorder=1)
        ax.gridlines(draw_labels=False, linewidth=0.25, color='gray', 
                    alpha=0.7, linestyle='--', zorder=6)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(title)
    
    # Plot difference map
    diff_plot = ax_diff.pcolormesh(lon, lat, diffs,
                                  vmin=diff_vmin, vmax=diff_vmax,
                                  transform=ccrs.PlateCarree(),
                                  cmap=diff_cmap,
                                  rasterized=True)
    
    ax_diff.coastlines(resolution='50m', linewidth=0.15, color='black', zorder=10)
    ax_diff.add_feature(cfeature.LAND, color='0.95', zorder=0)
    ax_diff.add_feature(cfeature.LAKES, color='grey', zorder=1)
    ax_diff.set_extent(extent, crs=ccrs.PlateCarree())
    ax_diff.set_title('(c) Difference')
    
    # Add colorbars
    cbar_ax1 = fig.add_axes([0.24, 0.04, 0.3, 0.04])
    cbar_ax2 = fig.add_axes([0.68, 0.04, 0.2, 0.04])
    
    fig.colorbar(thickness_plots[-1], cax=cbar_ax1, 
                label=f'{var_label} [m]',
                orientation='horizontal',
                extend='max')
    fig.colorbar(diff_plot, cax=cbar_ax2,
                label=f'{var_label} [m; difference]',
                orientation='horizontal',
                extend='both')
    
    plt.subplots_adjust(bottom=0.07, wspace=0.02)

    print(f"DEBUG: saving to {save_dir+'/GPSat_'+out_str+'_cmax'+str(cmax_main)+'.png'}")
    plt.savefig(save_dir+'/GPSat_'+out_str+'_cmax'+str(cmax_main)+'.png', dpi=300, facecolor="white", bbox_inches='tight')
    return


def plot_expert_usage(dfs, save_dir, config, IS2_grid=None):
    """
    Plot which expert locations were used vs skipped.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary of DataFrames from get_results_from_h5file
    save_dir : str
        Directory to save the plot
    config : dict
        Configuration dictionary
    IS2_grid : xarray.Dataset, optional
        IS2 grid for coordinate conversion (not used currently)
    """
    # Find relevant tables
    expert_table = None
    for table_name in ['expert_locs_SMOOTHED', 'expert_locs']:
        if table_name in dfs:
            expert_table = table_name
            break
    
    pred_table = None
    for table_name in ['preds_SMOOTHED', 'preds']:
        if table_name in dfs:
            pred_table = table_name
            break
    
    if not expert_table:
        print("No expert locations table found for plotting")
        return
    
    expert_locs = dfs[expert_table].copy()
    
    # Get coordinate columns
    coord_cols = [c for c in expert_locs.columns if c in ['x', 'y', 'time']]
    if not coord_cols:
        print("No coordinate columns found in expert locations")
        return
    
    # Determine which experts were used (made predictions)
    expert_locs['used'] = False
    expert_locs['has_valid_pred'] = False
    
    if pred_table:
        preds = dfs[pred_table]
        # Get unique expert locations that made valid predictions
        if 'f*' in preds.columns:
            valid_preds = preds[preds['f*'].notna()]
            if len(valid_preds) > 0:
                pred_coord_cols = [c for c in preds.columns if c in ['x', 'y', 'time']]
                if pred_coord_cols:
                    # Get unique expert locations with valid predictions
                    valid_expert_coords = valid_preds[pred_coord_cols].drop_duplicates()
                    valid_expert_set = set(tuple(row) for row in valid_expert_coords.values)
                    
                    # Mark experts that were used
                    expert_coords = expert_locs[coord_cols].values
                    for idx, coords in enumerate(expert_coords):
                        if tuple(coords) in valid_expert_set:
                            expert_locs.loc[expert_locs.index[idx], 'used'] = True
                            expert_locs.loc[expert_locs.index[idx], 'has_valid_pred'] = True
        
        # Also check for any predictions (even NaN) - means expert was processed
        pred_coord_cols = [c for c in preds.columns if c in ['x', 'y', 'time']]
        if pred_coord_cols:
            all_expert_coords = preds[pred_coord_cols].drop_duplicates()
            all_expert_set = set(tuple(row) for row in all_expert_coords.values)
            
            expert_coords = expert_locs[coord_cols].values
            for idx, coords in enumerate(expert_coords):
                if tuple(coords) in all_expert_set:
                    expert_locs.loc[expert_locs.index[idx], 'used'] = True
    
    # Convert x, y to lon, lat for plotting
    if 'lon' not in expert_locs.columns or 'lat' not in expert_locs.columns:
        # Convert from EASE2 to WGS84
        lon, lat = EASE2toWGS84(expert_locs['x'].values, expert_locs['y'].values, 
                                lat_0=config.get('lat_0', 90), lon_0=config.get('lon_0', 0))
        expert_locs['lon'] = lon
        expert_locs['lat'] = lat
    
    # Create plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-45, 90))
    
    # Plot skipped experts (not used)
    skipped = expert_locs[~expert_locs['used']]
    if len(skipped) > 0:
        ax.scatter(skipped['lon'], skipped['lat'],
                  c='red', s=15, alpha=0.6, marker='x',
                  transform=ccrs.PlateCarree(),
                  label=f'Skipped ({len(skipped)})',
                  zorder=5)
    
    # Plot used experts with valid predictions
    used_valid = expert_locs[expert_locs['has_valid_pred']]
    if len(used_valid) > 0:
        ax.scatter(used_valid['lon'], used_valid['lat'],
                  c='green', s=20, alpha=0.8, marker='o',
                  transform=ccrs.PlateCarree(),
                  label=f'Used with valid predictions ({len(used_valid)})',
                  zorder=6, edgecolors='darkgreen', linewidths=0.5)
    
    # Plot used experts with NaN predictions (processed but no data)
    used_nan = expert_locs[expert_locs['used'] & ~expert_locs['has_valid_pred']]
    if len(used_nan) > 0:
        ax.scatter(used_nan['lon'], used_nan['lat'],
                  c='orange', s=15, alpha=0.6, marker='s',
                  transform=ccrs.PlateCarree(),
                  label=f'Used but no valid predictions ({len(used_nan)})',
                  zorder=5)
    
    # Add map features
    ax.coastlines(resolution='50m', linewidth=0.15, color='black', zorder=10)
    ax.add_feature(cfeature.LAND, color='0.95', zorder=0)
    ax.add_feature(cfeature.LAKES, color='grey', zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.25, color='gray', 
                alpha=0.7, linestyle='--', zorder=6)
    ax.set_extent([-180, 180, 35, 90], crs=ccrs.PlateCarree())
    
    # Calculate statistics
    total = len(expert_locs)
    used = expert_locs['used'].sum()
    valid = expert_locs['has_valid_pred'].sum()
    skipped_count = total - used
    
    ax.set_title(f"Expert Location Usage\n"
                f"Total: {total} | Used: {used} ({100*used/total:.1f}%) | "
                f"Valid: {valid} ({100*valid/total:.1f}%) | Skipped: {skipped_count}",
                fontsize=12)
    
    ax.legend(fontsize=10, loc='center left', framealpha=0.9)
    plt.savefig(os.path.join(save_dir, 'expert_location_usage.png'), 
                dpi=300, facecolor="white", bbox_inches='tight')
    plt.close()
    print(f"Saved expert location usage plot to {os.path.join(save_dir, 'expert_location_usage.png')}")


def plot_thickness_std(array, target_date, figsize=(9.6, 8), out_str='', save_dir='./results/', variable='thickness'):
    """
    Create a plot of thickness/freeboard/snow_depth standard deviation.
    For freeboard and snow_depth: colorbar vmax=0.3 m.
    """
    var_label = variable.replace('_', ' ').title() if variable in ('freeboard', 'snow_depth') else 'Thickness'
    vmax_std = 0.3 if variable in ('freeboard', 'snow_depth') else 3.0
    
    # Create figure and subplot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(-45, 90))
    
    # Plot settings
    extent = [-180, 180, 55, 90]
    
    # Plot standard deviation
    p = ax.pcolormesh(array.lon, array.lat, array,
                     vmin=0., vmax=vmax_std,
                     transform=ccrs.PlateCarree(),
                     cmap='turbo',
                     rasterized=True)
    
    # Add map features
    ax.coastlines(resolution='50m', linewidth=0.15, color='black', zorder=10)
    ax.add_feature(cfeature.LAND, color='0.95', zorder=0)
    ax.add_feature(cfeature.LAKES, color='grey', zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.25, color='gray', 
                alpha=0.7, linestyle='--', zorder=6)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(f'Standard deviation of GPSat {var_label}')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.04])
    fig.colorbar(p, cax=cbar_ax, 
                label='Standard Deviation [m]',
                orientation='horizontal',
                extend='max')
    
    plt.subplots_adjust(bottom=0.1)
    print(f"DEBUG: saving to {save_dir+'/GPSat_'+variable+'_std_'+out_str+'.png'}")
    plt.savefig(save_dir+'/GPSat_'+variable+'_std_'+out_str+'.png', dpi=300, facecolor="white", bbox_inches='tight')
    return



def _variable_to_val_col(variable):
    """Map variable name (for paths) to data column name in dataset. Dataset uses: ice_thickness, freeboard (ICESat-2 total freeboard), snow_depth (NESOSIM)."""
    if variable == 'thickness':
        return 'ice_thickness'
    if variable == 'freeboard':
        return 'freeboard'  # ICESat-2 total freeboard in dataset
    if variable == 'snow_depth':
        return 'snow_depth'  # NESOSIM snow depth in dataset
    return variable  # use as-is


def main(num_days,
         plus_smap=False,
         plus_sic=False,
         config_overrides=None,
         target_date=None,
         output_results_path=None,
         smoothed_params_file_path=None,
         dataset_version=None,
         variable='thickness',
         smap_use_prediction_day_only=False,
         sic_use_prediction_day_only=True,
         is2_pref=False):
    
    print("\n=== Set up timing ===")
    
    ### Start timing
    script_start_time = time.time()
    step_times = {}
    def log_step(step_name, start, end):
        elapsed = end - start
        mins = elapsed / 60
        hours = mins / 60
        if mins < 60:
            step_times[step_name] = f"{mins:.2f} min"
        else:
            step_times[step_name] = f"{hours:.2f} hr ({mins:.2f} min)"
    
    ### Configuration
    print("\n1. Setting up configuration...")
    step_start = time.time()
    
    val_col = _variable_to_val_col(variable)
    
    # Create base output string - always use "run"
    base_out_str = "run"
    
    if plus_smap:
        out_str = f"{base_out_str}_{num_days}days_smap"
    elif plus_sic:
        out_str = f"{base_out_str}_{num_days}days_sic"
    else:
        out_str = f"{base_out_str}_{num_days}days_nosmap"
    
    print(out_str)
    print(f"Variable: {variable} -> val_col: {val_col}")
    
    # Loading is triggered solely by providing a smoothed_params_file_path
    is_loading = bool(smoothed_params_file_path)

    config = {
        'target_date': target_date,
        'num_days_before_after': int(num_days),
        'beam': ['bnum3'],
        'variable': variable,
        'val_col': val_col,
        'sic_cutoff': 0.15,
        'sic_coarsen_factor': 2,
        'sic_base_path': '/panfs/ccds02/home/aapetty/nobackup_symlink/Data/ICECONC/CDR/daily/final/v6',  # local CDR daily SIC
        'sic_use_s3_fallback': False,  # use local only; set True to fall back to NOAA CDR on S3 when local file not found
        'sic_use_prediction_day_only': sic_use_prediction_day_only,
        'smap_thickness_min': 0.0,
        'smap_thickness_max': 0.5,
        'smap_coarsen_factor': 4,
        'smap_use_prediction_day_only': smap_use_prediction_day_only,
        'smap_apply_region_filter': True,
        'smap_exclude_regions': [1],  # Central Arctic (IS2SITMOGR4 region 1); no SMAP used there
        'smap_cache_dir': '/explore/nobackup/people/aapetty/SMAP/thickness_cache/',
        'is2_alongtrack_base_path': '/explore/nobackup/people/aapetty/IS2thickness/rel007/run_v4/final_data_along_track/002/',
        'is2_gridded_base_path': '/explore/nobackup/people/aapetty/IS2thickness/rel007/run_v4/final_data_gridded/',
        'N_subsample': 1,
        'noise_std': 0.3,
        'expert_spacing': 200_000,
        'training_radius': 400_000,
        'inference_radius': 500_000,
        'model_type': 'GPflowSGPRModel', #oi_modes = ['GPflowGPRModel', 'GPflowSGPRModel', 'GPflowSVGPModel', 'sklearnGPRModel', 'GPflowVFFModel', 'GPflowASVGPModel']
        'pred_spacing': 25_000,
        'max_iter': 10000,
        'out_path': out_str,
        'use_region_filter': True,
        'filter_expert_locations': True,
        'filter_prediction_locations': True,
        'allowed_regions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'is2_pref': is2_pref,  # when True: prefer IS-2 over SMAP/SIC (drop SMAP/SIC points within is2_pref_radius of any IS-2 on same day; SIT_input per day use IS-2 where valid else SMAP/SIC)
        'is2_pref_radius': 10_000,  # meters; drop SMAP (or SIC) point if any IS-2 along-track point on same day is within this distance (SMAP is coarsened before this check)
        #- when a path is provided and differs from the new save_dir, the h5 config file is copied into the new versioned directory and the config path updated.
        'smoothed_params_file_path': smoothed_params_file_path,
        'base_results_dir': output_results_path if output_results_path else './results',
    }

    # No override merging: options must be provided explicitly via CLI args

    # Add load_sm_params indicator if loading smoothed parameters
    if is_loading:
        out_str = f"{out_str}_load_params"
        config['out_path'] = out_str
        print(f"Updated out_str to include load_sm_params indicator: {out_str}")

        
        # Defer automatic derivation of smoothed_params_file_path until after versioned directory creation
        # (No derivation in path-based loading mode)

    log_step("Configuration", step_start, time.time())

    
    # Create directory if it doesn't exist
    step_start = time.time()
    print("2. Creating output directory...")
    
    # Create results directory based on target_date (convert to YYYYMMDD)
    base_results_dir = config['base_results_dir']
    print(f"DEBUG: base_results_dir = {base_results_dir}")
    print(f"DEBUG: out_str = {out_str}")
    print(f"DEBUG: dataset_version = {dataset_version}")
    print(f"DEBUG: variable = {variable}")

    folder_date_str = target_date.replace('-', '')
    print(f"DEBUG: target_date = {target_date} -> folder_date_str = {folder_date_str}")

    # Build the new directory structure: {base_results_dir}/{dataset_version}/{variable}/{out_str}_{date}_{version}/
    if dataset_version:
        version_dir = os.path.join(base_results_dir, dataset_version)
        versioned_base_dir = os.path.join(base_results_dir, dataset_version, variable)
        
        # Create/update simple version-level config CSV
        version_config_path = os.path.join(version_dir, 'version_config.csv')
        config_exists = os.path.exists(version_config_path)
        
        # Create or update version config CSV
        try:
            os.makedirs(version_dir, exist_ok=True)
            if not config_exists:
                # Create new config file
                config_df = pd.DataFrame({
                    'setting': ['dataset_version', 'variable', 'num_days', 'smap_enabled', 'output_results_path', 'created', 'last_updated'],
                    'value': [dataset_version, variable, str(int(num_days)), str(plus_smap), 
                             output_results_path if output_results_path else './results',
                             datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                })
                config_df.to_csv(version_config_path, index=False)
                print(f"Created version config: {version_config_path}")
            else:
                # Update existing config (only update last_updated if values match, otherwise update values)
                config_df = pd.read_csv(version_config_path)
                config_dict = dict(zip(config_df['setting'], config_df['value']))
                
                # Update values
                config_dict['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if config_dict.get('num_days') != str(int(num_days)):
                    config_dict['num_days'] = str(int(num_days))
                if config_dict.get('smap_enabled') != str(plus_smap):
                    config_dict['smap_enabled'] = str(plus_smap)
                
                # Rebuild DataFrame
                updated_df = pd.DataFrame({
                    'setting': ['dataset_version', 'variable', 'num_days', 'smap_enabled', 'output_results_path', 'created', 'last_updated'],
                    'value': [config_dict.get('dataset_version', dataset_version),
                             config_dict.get('variable', variable),
                             config_dict.get('num_days', str(int(num_days))),
                             config_dict.get('smap_enabled', str(plus_smap)),
                             config_dict.get('output_results_path', output_results_path if output_results_path else './results'),
                             config_dict.get('created', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                             config_dict['last_updated']]
                })
                updated_df.to_csv(version_config_path, index=False)
                print(f"Updated version config: {version_config_path}")
        except Exception as e:
            print(f"WARNING: Could not write version config: {e}")
    else:
        # Fallback to old structure if dataset_version not provided
        versioned_base_dir = base_results_dir
    
    # Ensure the versioned base directory exists
    try:
        os.makedirs(versioned_base_dir, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create versioned base directory '{versioned_base_dir}': {e}")
        sys.exit(2)

    prefix = f"{out_str}_{folder_date_str}"
    try:
        existing = [
            d for d in os.listdir(versioned_base_dir)
            if os.path.isdir(os.path.join(versioned_base_dir, d)) and d.startswith(f"{prefix}_")
        ]
    except FileNotFoundError:
        existing = []
    print(f"DEBUG: Found {len(existing)} existing run dirs with prefix '{prefix}_'")
    if existing:
        print(f"DEBUG: Existing dirs: {sorted(existing)[-5:]}")

    # Always use v01 (overwrite on re-run)
    version_str = "v01"
    print(f"DEBUG: Using fixed version = {version_str} (overwrites existing)")
    # # Find the highest version number (commented out: use v02, v03, ... on re-run)
    # version = 1
    # version_pattern = re.compile(rf"^{re.escape(out_str)}_{folder_date_str}_v(\d{{2}})$")
    # for d in existing:
    #     m = version_pattern.match(d)
    #     if m:
    #         v = int(m.group(1))
    #         if v >= version:
    #             version = v + 1
    # version_str = f"v{version:02d}"
    # print(f"DEBUG: Selected version = {version_str}")

    # Create the final directory name with out_str before date and version at the end
    dir_name = f"{prefix}_{version_str}"
    save_dir = os.path.join(versioned_base_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created results directory: {save_dir}")

    # Ensure later saves using base_results_dir + out_path resolve to save_dir
    # Store the full relative path from base_results_dir
    if dataset_version:
        config['out_path'] = os.path.join(dataset_version, variable, dir_name)
    else:
        config['out_path'] = dir_name
    derived_dir = os.path.join(base_results_dir, config['out_path'])
    print(f"DEBUG: derived_dir (base_results_dir + out_path) = {derived_dir}")
    if os.path.abspath(save_dir) != os.path.abspath(derived_dir):
        print("WARNING: save_dir and derived_dir differ. Forcing config['out_path'] to match save_dir.")
        if dataset_version:
            config['out_path'] = os.path.join(dataset_version, variable, os.path.basename(save_dir))
        else:
            config['out_path'] = os.path.basename(save_dir)

    print(f"DEBUG: All subsequent saves should go to: {save_dir}")
    
    # Save config to CSV (run-specific config)
    config_df = pd.DataFrame(list(config.items()), columns=['Parameter', 'Value'])
    run_config_path = os.path.join(save_dir, 'config.csv')
    config_df.to_csv(run_config_path, index=False)
    print(f"Saved run config: {run_config_path}")

    # If loading smoothed params, copy required artifacts (h5, SMOOTHED json, source config)
    if is_loading:
        try:
            src_h5_path = config.get('smoothed_params_file_path')
            if not src_h5_path or not os.path.isfile(src_h5_path):
                print(f"ERROR: Source H5 file not found or not provided: {src_h5_path}")
                sys.exit(2)

            # Copy only the H5 file into the new run directory
            dest_h5_path = os.path.join(save_dir, os.path.basename(src_h5_path))
            if os.path.abspath(src_h5_path) != os.path.abspath(dest_h5_path):
                print(f"Copying H5 file to run directory:\n  from: {src_h5_path}\n  to:   {dest_h5_path}")
                shutil.copy2(src_h5_path, dest_h5_path)
                print(f"Copied H5 file to {dest_h5_path}")
            else:
                print("H5 file already in destination directory; no copy needed.")

            # Update config to point to the copy in the new run directory
            config['smoothed_params_file_path'] = dest_h5_path

        except Exception as e:
            print(f"ERROR copying smoothed parameter H5: {e}")
            print(traceback.format_exc())
            sys.exit(2)

    log_step("Create output directory & save config", step_start, time.time())

    # Validate availability of smoothed params when requested; if missing, exit
    if is_loading:
        fp = config.get('smoothed_params_file_path')
        if not (fp and os.path.isfile(fp)):
            print("ERROR: Loading requested but no valid smoothed_params_file_path found.")
            sys.exit(2)
    
    ### Initialize S3 filesystem
    step_start = time.time()
    print("3. Initializing S3 filesystem...")
    s3 = s3fs.S3FileSystem(anon=True)
    log_step("Initialize S3 filesystem", step_start, time.time())
    
    ### Read in the monthly gridded IS2 data. We use this for generating the grid/region mask. Maybe not very efficient..
    step_start = time.time()
    print("4. Reading IS2 data...")
    IS2 = read_IS2SITMOGR4(data_type='netcdf-local', version='004', 
                           local_data_path=config.get('is2_gridded_base_path')).rename({'longitude':'lon','latitude':'lat'})
    log_step("Read IS2 data", step_start, time.time())

    ### Create IS2 grid
    step_start = time.time()
    print("5. Creating IS2 grid...")
    IS2_grid = xr.Dataset(coords={'x':IS2.x,'y':IS2.y,'lat':IS2.lat,'lon':IS2.lon})
    x_mesh, y_mesh = np.meshgrid(IS2.x.values, IS2.y.values)
    log_step("Create IS2 grid", step_start, time.time())

    ### Set up parameters
    step_start = time.time()
    print("6. Setting up parameters and expert/prediction locations...")
    lat_0 = 90
    lon_0 = -45
    expert_x_range = [IS2.x.min(), IS2.x.max()]
    expert_y_range = [IS2.y.min(), IS2.y.max()]
    extent = [-180, 180, 35, 90]
    projection = "north"
    # Use a coarsened grid based on the IS2 grid for the experts
    IS2_coarse = IS2.sel(x=np.arange(IS2.x.min(),IS2.x.max(),config['expert_spacing']),y=np.arange(IS2.y.min(),IS2.y.max(),config['expert_spacing'])
                        ).sel(x=slice(expert_x_range[0],expert_x_range[1]),y=slice(expert_y_range[0],expert_y_range[1]))
    
    
    eloc_time = xr.date_range(config['target_date'],config['target_date'],freq='1D').to_numpy().astype("datetime64[D]").astype(float)
    eloc = xr.Dataset(coords={'x':IS2_coarse.x.drop_vars(['time'],errors='ignore')
                            ,'y':IS2_coarse.y.drop_vars(['time'],errors='ignore')
                            ,'lat':IS2_coarse.lat.drop_vars(['time','time_spatial'],errors='ignore')
                            ,'lon':IS2_coarse.lon.drop_vars(['time','time_spatial'],errors='ignore')
                            ,'time':eloc_time
                            }).to_dataframe().reset_index()
    ploc = IS2_grid.sel(x=slice(expert_x_range[0],expert_x_range[1]),y=slice(expert_y_range[1],expert_y_range[0])).to_dataframe().reset_index()
    ploc["is_in_ocean"] = globe.is_ocean(ploc['lat'], ploc['lon'])
    ploc = ploc.loc[ploc["is_in_ocean"]]
    
    ### Apply region filtering if enabled
    if config['use_region_filter']:
        print("6.1. Applying region filtering...")
        print(f"DEBUG: use_region_filter = {config['use_region_filter']}")
        print(f"DEBUG: filter_expert_locations = {config['filter_expert_locations']}")
        print(f"DEBUG: filter_prediction_locations = {config['filter_prediction_locations']}")
        print(f"DEBUG: allowed_regions = {config['allowed_regions']}")
        
        try:
            print_region_info()  # Print region information for reference
            print("DEBUG: print_region_info() completed successfully")
        except Exception as e:
            print(f"DEBUG: Error in print_region_info(): {str(e)}")
            print(f"DEBUG: Error type: {type(e)}")
            print("DEBUG: Continuing without region info...")
        
        # Get region mask for the target date (or closest available)
        print("DEBUG: Checking for region_mask in IS2 data...")
        print(f"DEBUG: IS2 data_vars: {list(IS2.data_vars.keys())}")
        
        if 'region_mask' in IS2.data_vars:
            print("DEBUG: region_mask found in IS2 data")
            try:
                # Use the region_mask from IS2 data
                region_mask_data = IS2.region_mask.sel(time=config['target_date'], method='nearest')
                print(f"DEBUG: Successfully selected region_mask for date: {config['target_date']}")
                print(f"DEBUG: region_mask_data shape: {region_mask_data.shape}")
                print(f"DEBUG: region_mask_data coords: {list(region_mask_data.coords.keys())}")
                print(f"DEBUG: region_mask_data values range: {region_mask_data.values}")
                print(f"DEBUG: region_mask_data unique values: {sorted(np.unique(region_mask_data.values))}")
                print(f"DEBUG: region_mask_data data type: {region_mask_data.dtype}")
            except Exception as e:
                print(f"DEBUG: Error selecting region_mask: {str(e)}")
                print(f"DEBUG: Error type: {type(e)}")
                print("DEBUG: Available times in region_mask:", IS2.region_mask.time.values if 'time' in IS2.region_mask.coords else "No time dimension")
                print("Warning: region_mask not found in IS2 data. Skipping region filtering.")
                config['use_region_filter'] = False
        else:
            print("DEBUG: region_mask not found in IS2 data_vars")
            print("Warning: region_mask not found in IS2 data. Skipping region filtering.")
            config['use_region_filter'] = False
        
        if config['use_region_filter']:
            print("DEBUG: Proceeding with region filtering...")
            
            # Filter expert locations
            if config['filter_expert_locations']:
                print(f"DEBUG: Filtering expert locations to regions: {config['allowed_regions']}")
                print(f"DEBUG: Expert locations before filtering: {len(eloc)}")
                print(f"DEBUG: Expert locations columns: {list(eloc.columns)}")
                print(f"DEBUG: Expert locations x range: {eloc['x'].min()} to {eloc['x'].max()}")
                print(f"DEBUG: Expert locations y range: {eloc['y'].min()} to {eloc['y'].max()}")
                
                try:
                    # Interpolate region_mask to expert locations (they may not be on the same grid)
                    
                    print("DEBUG: Imported scipy.interpolate.griddata successfully")
                    
                    print("DEBUG: Preparing data for griddata...")
                    print(f"DEBUG: IS2.x shape: {IS2.x.shape}")
                    print(f"DEBUG: IS2.y shape: {IS2.y.shape}")
                    print(f"DEBUG: region_mask_data shape: {region_mask_data.shape}")
                    
                    # Create meshgrid to ensure x and y coordinates are properly aligned
                    x_mesh, y_mesh = np.meshgrid(IS2.x.values, IS2.y.values)
                    print(f"DEBUG: x_mesh shape: {x_mesh.shape}")
                    print(f"DEBUG: y_mesh shape: {y_mesh.shape}")
                    
                    # Flatten the coordinates and values
                    x_coords = x_mesh.flatten()
                    y_coords = y_mesh.flatten()
                    region_values = region_mask_data.values.flatten()
                    
                    print(f"DEBUG: Flattened x_coords shape: {x_coords.shape}")
                    print(f"DEBUG: Flattened y_coords shape: {y_coords.shape}")
                    print(f"DEBUG: Flattened region_values shape: {region_values.shape}")
                    print(f"DEBUG: Region values range: {np.min(region_values)} to {np.max(region_values)}")
                    
                    # Verify all arrays have the same length
                    if len(x_coords) != len(y_coords) or len(x_coords) != len(region_values):
                        print(f"ERROR: Array length mismatch - x: {len(x_coords)}, y: {len(y_coords)}, region: {len(region_values)}")
                        raise ValueError("Coordinate arrays have different lengths")
                    
                    # Create points for interpolation
                    points = np.column_stack((x_coords, y_coords))
                    print(f"DEBUG: Points array shape: {points.shape}")
                    
                    # Get expert location coordinates
                    expert_points = np.column_stack((eloc['x'].values, eloc['y'].values))
                    print(f"DEBUG: Expert points shape: {expert_points.shape}")
                    
                    print("DEBUG: Calling griddata...")
                    expert_regions = griddata(points, region_values, expert_points, method='nearest')
                    print("DEBUG: griddata completed successfully")
                    print(f"DEBUG: expert_regions shape: {expert_regions.shape}")
                    print(f"DEBUG: expert_regions range: {np.min(expert_regions)} to {np.max(expert_regions)}")
                    
                    eloc['region'] = expert_regions
                    print("DEBUG: Added region column to eloc")
                    
                    # Convert region values to integers
                    eloc['region'] = eloc['region'].astype(int)
                    print("DEBUG: Converted expert regions to integers")
                    
                    # Count locations in each region
                    region_counts = eloc['region'].value_counts()
                    print(f"DEBUG: Region counts before filtering: {region_counts.to_dict()}")
                    
                    eloc = eloc[eloc['region'].isin(config['allowed_regions'])]
                    print(f"DEBUG: Expert locations after region filtering: {len(eloc)}")
                    
                    if len(eloc) > 0:
                        region_counts_after = eloc['region'].value_counts()
                        print(f"DEBUG: Region counts after filtering: {region_counts_after.to_dict()}")
                    else:
                        print("WARNING: No expert locations remain after region filtering!")
                        
                except Exception as e:
                    print(f"DEBUG: Error in expert location filtering: {str(e)}")
                    print(f"DEBUG: Error type: {type(e)}")
                    print("DEBUG: Full traceback:")
                    import traceback
                    print(traceback.format_exc())
                    print("WARNING: Skipping expert location filtering due to error")
            
            # Filter prediction locations
            if config['filter_prediction_locations']:
                print(f"DEBUG: Filtering prediction locations to regions: {config['allowed_regions']}")
                print(f"DEBUG: Prediction locations before filtering: {len(ploc)}")
                print(f"DEBUG: Prediction locations columns: {list(ploc.columns)}")
                
                try:
                    # Since prediction locations are on the same 25km grid as IS2 data, use direct indexing
                    # ploc is created from IS2_grid which has the same x,y coordinates as region_mask
                    # We can use direct indexing for much faster access
                    
                    print("DEBUG: Using griddata interpolation for prediction locations (same as expert locations)...")
                    print(f"DEBUG: ploc['x'] shape: {ploc['x'].shape}")
                    print(f"DEBUG: ploc['y'] shape: {ploc['y'].shape}")
                    print(f"DEBUG: ploc['x'] range: {ploc['x'].min()} to {ploc['x'].max()}")
                    print(f"DEBUG: ploc['y'] range: {ploc['y'].min()} to {ploc['y'].max()}")
                    
                    # Use the same griddata approach as expert locations
                    print("DEBUG: Imported scipy.interpolate.griddata successfully")
                    
                    print("DEBUG: Preparing data for griddata...")
                    print(f"DEBUG: IS2.x shape: {IS2.x.shape}")
                    print(f"DEBUG: IS2.y shape: {IS2.y.shape}")
                    print(f"DEBUG: region_mask_data shape: {region_mask_data.shape}")
                    
                    # Create meshgrid to ensure x and y coordinates are properly aligned
                    x_mesh, y_mesh = np.meshgrid(IS2.x.values, IS2.y.values)
                    print(f"DEBUG: x_mesh shape: {x_mesh.shape}")
                    print(f"DEBUG: y_mesh shape: {y_mesh.shape}")
                    
                    # Flatten the coordinates and values
                    x_coords = x_mesh.flatten()
                    y_coords = y_mesh.flatten()
                    region_values = region_mask_data.values.flatten()
                    
                    print(f"DEBUG: Flattened x_coords shape: {x_coords.shape}")
                    print(f"DEBUG: Flattened y_coords shape: {y_coords.shape}")
                    print(f"DEBUG: Flattened region_values shape: {region_values.shape}")
                    print(f"DEBUG: Region values range: {np.min(region_values)} to {np.max(region_values)}")
                    
                    # Verify all arrays have the same length
                    if len(x_coords) != len(y_coords) or len(x_coords) != len(region_values):
                        print(f"ERROR: Array length mismatch - x: {len(x_coords)}, y: {len(y_coords)}, region: {len(region_values)}")
                        raise ValueError("Coordinate arrays have different lengths")
                    
                    # Create points for interpolation
                    points = np.column_stack((x_coords, y_coords))
                    print(f"DEBUG: Points array shape: {points.shape}")
                    
                    # Get prediction location coordinates
                    pred_points = np.column_stack((ploc['x'].values, ploc['y'].values))
                    print(f"DEBUG: Prediction points shape: {pred_points.shape}")
                    
                    print("DEBUG: Calling griddata for prediction locations...")
                    pred_regions = griddata(points, region_values, pred_points, method='nearest')
                    print("DEBUG: griddata completed successfully for prediction locations")
                    print(f"DEBUG: pred_regions shape: {pred_regions.shape}")
                    print(f"DEBUG: pred_regions range: {np.min(pred_regions)} to {np.max(pred_regions)}")
                    
                    ploc['region'] = pred_regions
                    print("DEBUG: Added region column to ploc")
                    
                    # Convert region values to integers
                    ploc['region'] = ploc['region'].astype(int)
                    print("DEBUG: Converted prediction regions to integers")
                    
                    # Count locations in each region
                    region_counts = ploc['region'].value_counts()
                    print(f"DEBUG: Prediction region counts before filtering: {region_counts.to_dict()}")
                    print(f"DEBUG: Unique region values: {sorted(ploc['region'].unique())}")
                    print(f"DEBUG: Allowed regions: {config['allowed_regions']}")
                    
                    # Check if any regions match the allowed regions
                    matching_regions = ploc[ploc['region'].isin(config['allowed_regions'])]
                    print(f"DEBUG: Locations matching allowed regions: {len(matching_regions)}")
                    
                    ploc = ploc[ploc['region'].isin(config['allowed_regions'])]
                    print(f"DEBUG: Prediction locations after region filtering: {len(ploc)}")
                    
                    if len(ploc) > 0:
                        region_counts_after = ploc['region'].value_counts()
                        print(f"DEBUG: Prediction region counts after filtering: {region_counts_after.to_dict()}")
                    else:
                        print("WARNING: No prediction locations remain after region filtering!")
                        print("DEBUG: This means no prediction locations have region values in the allowed_regions list")
                        print("DEBUG: Consider checking if the region_mask data contains the expected region values")
                        print("DEBUG: Or temporarily disable region filtering by setting use_region_filter=False")
                        
                except Exception as e:
                    print(f"DEBUG: Error in prediction location filtering: {str(e)}")
                    print(f"DEBUG: Error type: {type(e)}")
                    print("DEBUG: Full traceback:")
                    import traceback
                    print(traceback.format_exc())
                    print("WARNING: Skipping prediction location filtering due to error")
        else:
            print("DEBUG: Region filtering disabled or failed, continuing without filtering")
    
    log_step("Set up parameters & expert/prediction locations", step_start, time.time())

    
    ### Plot prediction and expert locations
    step_start = time.time()
    print("6.2. Plotting prediction and expert locations...")
    print(f"DEBUG: Prediction locations after region filtering: {len(ploc)}")
    print(f"DEBUG: Expert locations after region filtering: {len(eloc)}")
    
    # Debug prediction locations data
    if len(ploc) > 0:
        print(f"DEBUG: ploc columns: {list(ploc.columns)}")
        print(f"DEBUG: ploc lon range: {ploc['lon'].min():.2f} to {ploc['lon'].max():.2f}")
        print(f"DEBUG: ploc lat range: {ploc['lat'].min():.2f} to {ploc['lat'].max():.2f}")
        print(f"DEBUG: ploc sample data:")
        print(ploc.head())
    else:
        print("ERROR: No prediction locations available for plotting!")
        print("DEBUG: This means region filtering removed all locations")
        print("DEBUG: Check the region filtering debug output above")
    
    # Debug expert locations data
    if len(eloc) > 0:
        print(f"DEBUG: eloc columns: {list(eloc.columns)}")
        print(f"DEBUG: eloc lon range: {eloc['lon'].min():.2f} to {eloc['lon'].max():.2f}")
        print(f"DEBUG: eloc lat range: {eloc['lat'].min():.2f} to {eloc['lat'].max():.2f}")
    else:
        print("ERROR: No expert locations available for plotting!")
    
    plot_prec_elocs = True
    if plot_prec_elocs:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-45,90))
        
        # Plot prediction locations as scatter points
        print(f"DEBUG: Plotting {len(ploc)} prediction locations")
        print(f"DEBUG: ploc lon has NaN: {ploc['lon'].isna().any()}")
        print(f"DEBUG: ploc lat has NaN: {ploc['lat'].isna().any()}")
        print(f"DEBUG: ploc lon finite: {ploc['lon'].notna().sum()}")
        print(f"DEBUG: ploc lat finite: {ploc['lat'].notna().sum()}")
        
        # Remove any NaN values for plotting
        ploc_clean = ploc.dropna(subset=['lon', 'lat'])
        print(f"DEBUG: After removing NaN, plotting {len(ploc_clean)} prediction locations")
        
        if len(ploc_clean) > 0:
            s1 = ax.scatter(ploc_clean['lon'],
                           ploc_clean['lat'],
                           c=ploc_clean['lat'],
                           cmap='viridis',
                           vmin=30, vmax=90,
                           transform=ccrs.PlateCarree(),
                           linewidth=0,
                           rasterized=True,
                           s=0.5,
                           label='prediction locations',
                           alpha=0.7)
            print("DEBUG: Prediction locations scatter plot created successfully")
        else:
            print("WARNING: No valid prediction locations to plot after removing NaN values")
        
        # Plot expert locations as scatter points
        print(f"DEBUG: Plotting {len(eloc)} expert locations")
        print(f"DEBUG: eloc lon has NaN: {eloc['lon'].isna().any()}")
        print(f"DEBUG: eloc lat has NaN: {eloc['lat'].isna().any()}")
        
        # Remove any NaN values for plotting
        eloc_clean = eloc.dropna(subset=['lon', 'lat'])
        print(f"DEBUG: After removing NaN, plotting {len(eloc_clean)} expert locations")
        
        if len(eloc_clean) > 0:
            s2 = ax.scatter(eloc_clean['lon'],
                           eloc_clean['lat'],
                           c=eloc_clean['lat'],
                           cmap='viridis',
                           vmin=30, vmax=90,
                           transform=ccrs.PlateCarree(),
                           linewidth=0,
                           rasterized=True,
                           s=10,
                           label='expert locations')
            print("DEBUG: Expert locations scatter plot created successfully")
        else:
            print("WARNING: No valid expert locations to plot after removing NaN values")
        
        # Add map features
        ax.coastlines(resolution='50m', linewidth=0.15, color='black', zorder=10)
        ax.add_feature(cfeature.LAND, color='0.95', zorder=0)
        ax.add_feature(cfeature.LAKES, color='grey', zorder=1)
        ax.gridlines(draw_labels=False, linewidth=0.25, color='gray', 
                    alpha=0.7, linestyle='--', zorder=6)
        ax.set_extent([-180, 180, 35, 90], crs=ccrs.PlateCarree())
        ax.set_title("Prediction vs. Expert locations (latitude values)")
        
        ax.legend(fontsize=15, loc='center left')
        plt.savefig(save_dir+'/prediction_expert_locations.png', dpi=300, facecolor="white", bbox_inches='tight')
        plt.close()
    log_step("Plot prediction/expert locations", step_start, time.time())

    
    ### Load data
    step_start = time.time()
    print("7. Loading data around target date...")
    along_track_data, along_track_data_gridded, smap_data, smap_data_gridded = load_data_around_target_date(
        config=config,
        IS2=IS2,
        plus_smap=plus_smap,
        plus_sic=plus_sic
    )
    log_step("Load data around target date", step_start, time.time())

    
    ### Process data
    step_start = time.time()
    print("8. Processing data...")
    print(f"SMAP/SIC configuration:")
    print(f"  - SMAP use prediction day only: {config.get('smap_use_prediction_day_only', False)}")
    if plus_sic:
        print(f"  - SIC use prediction day only: {config.get('sic_use_prediction_day_only', True)}")
    print(f"  - Apply region filter: {config.get('smap_apply_region_filter', False)}")
    print(f"  - Exclude SMAP in regions (IS2SITMOGR4 mask): {config.get('smap_exclude_regions', [])}")
    print(f"  - Thickness range: {config.get('smap_thickness_min', 0.0)}m to {config.get('smap_thickness_max', 0.5)}m")
    print(f"  - Coarsen factor: {config.get('smap_coarsen_factor', 1)}")
    
    val_col = config['val_col']
    if smap_data:
        smap_data_df = pd.concat(smap_data).dropna().reset_index()
        smap_data_df['time'] = smap_data_df.time.values.astype("datetime64[D]").astype(float)
        print('SMAP/SIC times:', smap_data_df.time.values)
        smap_data_df = smap_data_df.astype('float64')
        if 'index' in smap_data_df.columns:
            smap_data_df = smap_data_df.drop(columns=['index'])
        
        # Add small random noise to prevent exact duplicates
        np.random.seed(42)  # For reproducibility
        smap_data_df['x'] += np.random.normal(0, 1e-6, len(smap_data_df))
        smap_data_df['y'] += np.random.normal(0, 1e-6, len(smap_data_df))
        
        print(f"SMAP/SIC data summary after processing:")
        print(f"  - Total rows: {len(smap_data_df):,}")
        print(f"  - Date range: {smap_data_df['time'].min():.0f} to {smap_data_df['time'].max():.0f}")
        if val_col in smap_data_df.columns:
            print(f"  - {val_col} range: {smap_data_df[val_col].min():.3f}m to {smap_data_df[val_col].max():.3f}m")
    else:
        print("No SMAP/SIC data available")
        smap_data_df = pd.DataFrame()

    # Require SIC data when --sic true (SIC-only mode); exit with error if none loaded
    if plus_sic and not plus_smap and len(smap_data_df) == 0:
        print("ERROR: SIC was requested (--sic true) but no SIC data was loaded.")
        print("  Check: sic_base_path exists and contains YYYY/*YYYYMMDD*.nc files for the target date.")
        print("  sic_base_path: {}".format(config.get('sic_base_path')))
        sys.exit(1)

    along_track_data_df = along_track_data.to_dataframe().dropna().reset_index()
    along_track_data_df['time'] = along_track_data_df.time.values.astype("datetime64[D]").astype(float)
    along_track_data_df = along_track_data_df.astype('float64')
    along_track_data_df = along_track_data_df.drop(columns=['along_track_distance_section', 'lat', 'lon'])
    
    # Add small random noise to prevent exact duplicates
    np.random.seed(42)  # For reproducibility
    along_track_data_df['x'] += np.random.normal(0, 1e-6, len(along_track_data_df))
    along_track_data_df['y'] += np.random.normal(0, 1e-6, len(along_track_data_df))
    log_step("Process data", step_start, time.time())

    
    ### Combine dataframes (optionally prefer IS-2: drop SMAP/SIC points within radius of any IS-2 on same day)
    step_start = time.time()
    is2_pref = config.get('is2_pref', False)
    print("9. Combining dataframes" + (" (IS-2 preferred: drop SMAP/SIC where IS-2 nearby)" if is2_pref else "") + "...")
    if (plus_smap or plus_sic) and len(smap_data_df) > 0:
        if is2_pref:
            is2_pref_combine_start = time.time()
            # Data: IS-2 along-track (irregular ~10 km) and coarsened SMAP (or SIC) grid. Drop SMAP/SIC point if any IS-2 on same day is within radius (so we use IS-2 there).
            radius_m = config.get('is2_pref_radius', 25_000)
            along_track_data_df['_t'] = along_track_data_df['time'].astype('datetime64[D]')
            smap_data_df['_t'] = smap_data_df['time'].astype('datetime64[D]')
            keep_mask = np.ones(len(smap_data_df), dtype=bool)
            for day in smap_data_df['_t'].unique():
                is2_day = along_track_data_df.loc[along_track_data_df['_t'] == day, ['x', 'y']].values
                day_positions = np.where((smap_data_df['_t'] == day).values)[0]  # integer positions in smap_data_df
                smap_day_xy = smap_data_df.loc[smap_data_df['_t'] == day, ['x', 'y']].values
                if len(is2_day) == 0:
                    continue
                dist = cdist(smap_day_xy, is2_day)  # (n_smap_day, n_is2_day)
                min_dist = dist.min(axis=1)
                for i, pos in enumerate(day_positions):
                    if min_dist[i] <= radius_m:
                        keep_mask[pos] = False
            along_track_data_df.drop(columns=['_t'], inplace=True)
            smap_data_df.drop(columns=['_t'], inplace=True)
            smap_filtered = smap_data_df.loc[keep_mask].reset_index(drop=True)
            combined_df = pd.concat([along_track_data_df, smap_filtered], ignore_index=True)
            combined_df = combined_df.sort_values(by='time')
            n_dropped = len(smap_data_df) - len(smap_filtered)
            if n_dropped > 0:
                print(f"  Dropped {n_dropped} SMAP/SIC points with IS-2 within {radius_m/1e3:.0f} km same day (IS-2 preferred)")
            log_step("IS2_pref (combine: drop SMAP/SIC within radius)", is2_pref_combine_start, time.time())
        else:
            smap_filtered = smap_data_df
            combined_df = pd.concat([along_track_data_df, smap_data_df], ignore_index=True)
            combined_df = combined_df.sort_values(by='time')
    else:
        combined_df = along_track_data_df
        smap_filtered = pd.DataFrame()
    
    print(f"Combined DataFrame length: {len(combined_df)} rows")
    print(f"Along-track data: {len(along_track_data_df)} rows")
    if (plus_smap or plus_sic) and len(smap_data_df) > 0:
        print(f"SMAP/SIC data: {len(smap_filtered)} rows used" + (" (dropped where IS-2 within radius)" if is2_pref else ""))
    log_step("Combine dataframes", step_start, time.time())

    
    ### Set up data configuration
    step_start = time.time()
    print("10. Setting up data configuration...")
    data = {
        "data_source": combined_df, 
        "obs_col": config['val_col'],
        "coords_col": ["x", "y", "time"],
        "local_select": [
            {
                "col": "time",
                "comp": "<=",
                "val": config['num_days_before_after']
            },
            {
                "col": "time",
                "comp": ">=",
                "val": -1*config['num_days_before_after']
            },
            {
                "col": ["x", "y"],
                "comp": "<",
                "val": config['training_radius']
            }
        ]
    }
    log_step("Set up data configuration", step_start, time.time())
    
    ### Set up local expert configuration
    step_start = time.time()
    print("11. Setting up local expert locations...")
    local_expert = {
        "source": eloc
    }
    log_step("Set up local expert configuration", step_start, time.time())
    
    ### Set up prediction locations
    step_start = time.time()
    print("11.1. Setting up prediction locations...")
    pred_loc = {
        "method": "from_dataframe",
        "df": ploc,
        "max_dist": config['inference_radius']
    }
    log_step("Set up prediction locations", step_start, time.time())

    
    ### Set up model configuration
    step_start = time.time()
    print("12. Setting up model configuration...")
    
    # Base model configuration
    model = {
        "oi_model": config['model_type'],
        "init_params": {
            "likelihood_variance": config['noise_std']**2,
            "coords_scale": [25000, 25000, 1],
            "jitter": 1e-3
        },
        "optim_kwargs": {
            #"fixed_params": ['likelihood_variance'],#FIXING THE LIKLIHOOD FUNCTION TO INCREASE SPEED?
            "max_iter": config['max_iter']
        }
    }
    
    # Add constraints only if NOT loading smoothed parameters
    if not is_loading:
        print(f"DEBUG:adding constraints as not loaded parameters")
        model["constraints"] = {
            "lengthscales": {
                "low": [10000, 10000, 1.0],   # 10 km min for x, y (m); 1 day for time
                "high": [1000_000, 1000_000, 50]  # allow ~500–1000 km in central Arctic
            },
            "likelihood_variance": {
                "low": 1e-2,  # Increased minimum likelihood variance
                "high": 1.0   # Increased maximum likelihood variance
            },
            "kernel_variance": {
                "low": 1e-2,  # Increased minimum kernel variance
                "high": 10.0  # Increased maximum kernel variance
            }
        }
    
    # Add load_params if loading smoothed parameters
    if is_loading:
        print(f"DEBUG:adding load_params as loaded parameters")
        model["load_params"] = {
            "file": config['smoothed_params_file_path'],
            "param_names": ["likelihood_variance","kernel_variance","lengthscales"],
            "table_suffix": "_SMOOTHED"
        }
    
    log_step("Set up model configuration", step_start, time.time())

    # === OI, hyperparameter optimization, smoothing, and unsmoothed plotting ===
    print(f"DEBUG: is_loading = {is_loading}")
    print(f"DEBUG: Condition 'not is_loading' = {not is_loading}")
    
    if not is_loading:
        ### Initializing LocalExpertOI
        step_start = time.time()
        print("13. Initializing LocalExpertOI...")
        store_path = get_parent_path(config['base_results_dir']+"/"+config['out_path'], "IS2_interp_test_petty_v1.h5")
        print(f"DEBUG: store_path: {store_path}")
        try:
            locexp = LocalExpertOI(expert_loc_config=local_expert,
                                  data_config=data,
                                  model_config=model,
                                  pred_loc_config=pred_loc)
        except Exception as e:
            print(f"Error initializing LocalExpertOI: {str(e)}")
            raise
        log_step("Initialize LocalExpertOI", step_start, time.time())
        
        ### Run LocalExpertOI
        step_start = time.time()
        print("13.1 Running LocalExpertOI...")
        with HiddenPrints():
            try:
                locexp.run(
                    store_path=store_path,
                    optimise=True,
                    check_config_compatible=False
                )
                print("14.3. LocalExpertOI run completed successfully")
            except Exception as e:
                print(f"Error during LocalExpertOI run: {str(e)}")
                print(f"Error type: {type(e)}")
                print("Full traceback:")
                print(traceback.format_exc())
                raise
        log_step("Run LocalExpertOI", step_start, time.time())
        
        ### Extract results
        step_start = time.time()
        print("Extracting results...")
        try:
            dfs, oi_config = get_results_from_h5file(store_path)
            print(f"tables in results file: {list(dfs.keys())}")
            
            # Plot expert location usage
            try:
                plot_expert_usage(dfs, save_dir, config, IS2_grid)
            except Exception as e:
                print(f"Warning: Could not plot expert usage: {e}")
        except Exception as e:
            print(f"Error extracting results: {str(e)}")
            raise
        log_step("Extract results (unsmoothed)", step_start, time.time())
    
        ### Plot and save results
        step_start = time.time()
        print("13.2. Plotting and saving results...")
        try:
            # a template to be used for each created plot config
            plot_template = {
                "plot_type": "heatmap",
                "x_col": "x",
                "y_col": "y",
                # use a northern hemisphere projection, centered at (lat,lon) = (90,0)
                "subplot_kwargs": {"projection": projection},
                "lat_0": lat_0,
                "lon_0": lon_0,
                # any additional arguments for plot_hist
                "plot_kwargs": {
                    "scatter": False,
                },
                # lat/lon_col needed if scatter = True
                # TODO: remove the need for this
                "lat_col": "lat",
                "lon_col": "lon",
            }
            fig = plot_hyper_parameters(dfs,
                                        coords_col=oi_config[0]['data']['coords_col'],  # ['x', 'y', 'time']
                                        row_select=None,  # this could be used to select a specific date in results data
                                        table_names=["lengthscales", "kernel_variance", "likelihood_variance"],
                                        plot_template=plot_template,
                                        plots_per_row=3,
                                        suptitle="hyper params",
                                        qvmin=0.0001,
                                        qvmax=0.9999)
            plt.savefig(save_dir+'/GPSat_unsmoothed_weights_'+config['target_date']+'.png', dpi=300, facecolor="white", bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Error plotting results: {str(e)}")
            raise
        log_step("Plot and save results (unsmoothed)", step_start, time.time())
    
        ### Smooth hyperparameters
        step_start = time.time()
        print("13.3. Smoothing hyperparameters...")
        try:
            # Lengthscale bounds: table stores values in scaled space (physical / coords_scale).
            # Derive scaled min/max from model coords_scale so they stay correct if grid/scale changes.
            coords_scale = np.atleast_1d(model["init_params"]["coords_scale"]).flatten()
            ls_min_phys = [10_000, 10_000, 1.0]    # 10 km x,y; 1 day (physical)
            ls_max_phys = [500_000, 500_000, 50]  # 500 km x,y; 50 days (physical)
            ls_min_scaled = [ls_min_phys[i] / coords_scale[i] for i in range(min(3, len(coords_scale)))]
            ls_max_scaled = [ls_max_phys[i] / coords_scale[i] for i in range(min(3, len(coords_scale)))]
            smooth_config = {
            # get hyper parameters from the previously stored results
            "result_file": store_path,
            # store the smoothed hyper parameters in the same file
            "output_file": store_path,
            # get the hyper params from tables ending with this suffix ("" is default):
            "reference_table_suffix": "",
            # newly smoothed hyper parameters will be store in tables ending with table_suffix
            "table_suffix": "_SMOOTHED",
            # dimension names to smooth over
            "xy_dims": [
                "x",
                "y"
            ],
            # parameters to smooth
            "params_to_smooth": [
                "lengthscales",
                "kernel_variance"
                #"likelihood_variance"
            ],
            # length scales for kernel smoother (x, y in m; same units as expert grid). min/max in scaled space (from coords_scale above)
            "smooth_config_dict": {
                "lengthscales": {
                    "l_x": 300_000,
                    "l_y": 300_000,
                    "min": ls_min_scaled,
                    "max": ls_max_scaled
                },
                "kernel_variance": {
                    "l_x": 300_000,
                    "l_y": 300_000,
                    "max": 2.5
                }
            },
            "save_config_file": True
            }
            smooth_result_config_file = smooth_hyperparameters(**smooth_config)
            print(f"Smooth result config file: {smooth_result_config_file}")
            # modify the model configuration to include "load_params"

            model_smooth = model.copy()
            model_smooth["load_params"] = {
                "file": store_path,
                "table_suffix": smooth_config["table_suffix"]
            }
        except Exception as e:
            print(f"Error smoothing hyperparameters: {str(e)}")
            raise
        log_step("Smooth hyperparameters", step_start, time.time())
    
        ### Run LocalExpertOI with smoothed hyperparameters
        step_start = time.time()
        print("13.4. Running LocalExpertOI with smoothed hyperparameters")
        try:    
            locexp_smooth = LocalExpertOI(expert_loc_config=local_expert,
                                data_config=data,
                                model_config=model_smooth,
                                pred_loc_config=pred_loc)
            # run optimal interpolation (again)
            # - this time don't optimise hyper parameters, but make predictions
            # - store results in new tables ending with '_SMOOTHED'
            with HiddenPrints():
                locexp_smooth.run(store_path=store_path,
                                optimise=False,
                                predict=True,
                                table_suffix=smooth_config['table_suffix'],
                                check_config_compatible=False)
            print("13.4. LocalExpertOI run completed successfully")
            # extract, store in dict
            dfs, oi_config = get_results_from_h5file(store_path)
        except Exception as e:
            print(f"Error during LocalExpertOI run: {str(e)}")
            print(f"Error type: {type(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            raise
        log_step("Run LocalExpertOI with smoothed hyperparameters", step_start, time.time())
    
        ### Plot and save smoothed hyperparameters
        step_start = time.time()
        print("13.5 Plotting smoothed hyperparameters...")
        try:
            # plot and save results
            # Use full range (0.0/1.0) to avoid cut-offs, same as unsmoothed
            fig = plot_hyper_parameters(dfs,
                            coords_col=oi_config[0]['data']['coords_col'],  # ['x', 'y', 't']
                            row_select=None,
                            table_names=["lengthscales", "kernel_variance", "likelihood_variance"],
                            #table_suffix=smooth_config["table_suffix"],
                            table_suffix = '_SMOOTHED',
                            plot_template=plot_template,
                            plots_per_row=3,
                            suptitle="smoothed hyper params",
                            qvmin=0.0,  # Show full range (min)
                            qvmax=1.0)  # Show full range (max)
            plt.tight_layout()
            plt.savefig(save_dir+'/GPSat_smoothed_weights_'+config['target_date']+'.png', dpi=300, facecolor="white", bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Error plotting smoothed hyperparameters: {str(e)}")
            raise
        log_step("Plot and save smoothed hyperparameters", step_start, time.time())
    
    # === If loading smoothed parameters, skip to OI with loaded params ===
    print(f"DEBUG: About to check loading condition: {is_loading}")
    if is_loading:
        print("13. Running OI with loaded smoothed parameters...")
        
        #store_path = get_parent_path("results/"+config['out_path'], "IS2_interp_test_petty_v1.h5")
        store_path = config['smoothed_params_file_path']
        # MAYBE HERE WE SHOULD INSTEAD SET STORE_PATH TO smoothed_params_file_path
        print(f"DEBUG: store_path: {store_path}")
        
        print(f"DEBUG:running local expert oi with loaded parameters")
        # Create LocalExpertOI with loaded parameters - model already has load_params configured
        locexp_smooth = LocalExpertOI(expert_loc_config=local_expert,
                                     data_config=data,
                                     model_config=model,  # Use the model with load_params already set
                                     pred_loc_config=pred_loc)
        print(f"DEBUG: locexp_smooth: {locexp_smooth}")
        # Run OI with loaded parameters
        print(f"DEBUG: running locexp_smooth.run")
        locexp_smooth.run(store_path=store_path,
                        optimise=False,
                        predict=True,
                        table_suffix="_SMOOTHED",
                        check_config_compatible=False)
        print("13.1. LocalExpertOI run with loaded smoothed parameters completed successfully")
        
        # extract, store in dict
        dfs, oi_config = get_results_from_h5file(store_path)
        print(f"Available tables after OI run: {list(dfs.keys())}")
        
        # Plot expert location usage
        try:
            plot_expert_usage(dfs, save_dir, config, IS2_grid)
        except Exception as e:
            print(f"Warning: Could not plot expert usage: {e}")

    ### Generate netcdf file
    step_start = time.time()
    print('14. Generating netcdf file...')
    try:
        # plot and save results
        # Check which prediction table is available
        out_path_pred = "IS2_SMAP_GPSat_"+config['target_date']+".nc"
        extra_var=''
        if "preds_SMOOTHED" in dfs:
            plt_data = dfs["preds_SMOOTHED"]
            
        elif "preds" in dfs:
            plt_data = dfs["preds"]
            extra_var='unsmoothed'
        else:
            # List available tables for debugging
            print(f"Available tables in results: {list(dfs.keys())}")
            raise KeyError("No prediction table found. Available tables: " + str(list(dfs.keys())))
        weighted_values_kwargs = {
            "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_time"],
            "dist_to_col": ["x", "y", "time"],
            "val_cols": ["f*", "f*_var"],
            "weight_function": "gaussian",
            "lengthscale": config['inference_radius']/2
        }
        plt_data = get_weighted_values(df=plt_data, **weighted_values_kwargs)
        plt_data_indexed = plt_data.set_index(['pred_loc_time','pred_loc_y','pred_loc_x'])
        plt_data_xarray = plt_data_indexed.to_xarray().rename({'pred_loc_x':'x','pred_loc_y':'y','pred_loc_time':'time','f*':'SIT','f*_var':'SIT_var'})
        IS2_grid_subset = IS2_grid.sel(x=plt_data_xarray.x,y=plt_data_xarray.y).drop_vars('time',errors='ignore')
        lon_grid, lat_grid = IS2_grid_subset.lon,IS2_grid_subset.lat

        ####to use an equal area projection instead
        #val2d, x_grid, y_grid = dataframe_to_2d_array(df=plt_data,
        #                                  x_col='pred_loc_x',
        #                                  y_col='pred_loc_y',
        #                                  val_col='f*')
        #lon_grid, lat_grid = EASE2toWGS84(x_grid, y_grid, lat_0=lat_0, lon_0=lon_0)

        IS2_interp = plt_data_xarray.assign_coords({'lat':lat_grid,'lon':lon_grid})
        IS2_interp['time'] = IS2_interp.time.astype("datetime64[D]")

        #full array
        # why???
        IS2_interp = xr.broadcast(IS2_interp,IS2.isel(time=0))[0]
        IS2_interp['lat'],IS2_interp['lon'] = IS2['lat'],IS2['lon']

        #IS2_interp = IS2_interp.resample(time='1MS').mean()
        #IS2_interp = IS2_interp.where((~CDR_NH_sic_monthly.isnull()) & (CDR_NH_sic_monthly>0.5) & (IS2_interp.SIT>0))
        #IS2_interp = IS2_interp.where((~CDR_NH_sic_daily.isnull()) & (CDR_NH_sic_daily>0.5) & (IS2_interp.SIT>0))
        #IS2_interp = IS2_interp.where((IS2_interp.SIT>0))
        
        # ADD THE BINNED DATA TO THE DATASET (optionally prefer IS-2: per day use IS-2 where valid, else SMAP/SIC)
        along_track_data_gridded = along_track_data_gridded.sortby('time')
        # Resample the binned along-track input data to daily
        along_track_data_gridded_daily = along_track_data_gridded.resample(time='1D').mean().assign_coords({'lat':IS2.lat,'lon':IS2.lon})
        val_col = config['val_col']
        is2_pref = config.get('is2_pref', False)
        if (plus_smap or plus_sic) and smap_data_gridded:
            smap_data_gridded_daily = xr.concat(smap_data_gridded, dim='time').assign_coords({'lat':IS2.lat,'lon':IS2.lon})
            if is2_pref:
                is2_pref_sit_start = time.time()
                # Per day use IS-2 where valid, else SMAP/SIC (IS-2 preferred)
                smap_reindexed = smap_data_gridded_daily[val_col].reindex(time=along_track_data_gridded_daily.time, method='nearest')
                combined_daily = xr.where(
                    np.isfinite(along_track_data_gridded_daily[val_col].values),
                    along_track_data_gridded_daily[val_col].values,
                    smap_reindexed.values
                )
                combined_daily = xr.DataArray(
                    combined_daily,
                    dims=along_track_data_gridded_daily[val_col].dims,
                    coords=along_track_data_gridded_daily[val_col].coords
                )
                mean_input_data = combined_daily.mean(dim='time', skipna=True)
                log_step("IS2_pref (SIT_input: IS-2 where valid else SMAP/SIC)", is2_pref_sit_start, time.time())
            else:
                array2 = along_track_data_gridded_daily[val_col].mean(dim='time', skipna=True)
                array3 = smap_data_gridded_daily[val_col].mean(dim='time', skipna=True)
                combined = xr.concat([array2, array3], dim='time')
                mean_input_data = combined.mean(dim='time', skipna=True)
        else:
            mean_input_data = along_track_data_gridded_daily[val_col].mean(dim='time', skipna=True)
        IS2_interp['SIT_input'] = mean_input_data
        print(IS2_interp)
        

        store_path_nc = get_parent_path(config['base_results_dir']+"/"+config['out_path'], out_path_pred)
        print(f"DEBUG: saving to {store_path_nc}")
        IS2_interp.to_netcdf(store_path_nc)
    except Exception as e:
        print(f"Error generating netcdf file: {str(e)}")
        raise
    log_step("Generate netcdf file", step_start, time.time())
        
    ### Plot SIT and SIT_VAR
    step_start = time.time()
    print("15. Plotting SIT and SIT_VAR...")

    try:
        var = config.get('variable', 'thickness')
        # Main comparison: thickness 5 m (and thin-ice 1 m); freeboard/snow_depth 0.6 m only
        plot_thickness_comparison(IS2_interp, target_date=config['target_date'], out_str=extra_var+config['target_date'], days_str=str(config['num_days_before_after']), save_dir=save_dir, cmax=5, variable=var)
        if var == 'thickness':
            plot_thickness_comparison(IS2_interp, target_date=config['target_date'], out_str=extra_var+config['target_date'], days_str=str(config['num_days_before_after']), save_dir=save_dir, cmax=1, variable=var)
        # Plot standard deviation
        plot_thickness_std(IS2_interp.SIT_var[0], target_date=config['target_date'], out_str=extra_var+config['target_date'], save_dir=save_dir, variable=var)
    except Exception as e:
        print(f"Error plotting SIT and SIT_VAR: {str(e)}")
        raise
    log_step("Plot SIT and SIT_UNCERTAINTY", step_start, time.time())
    
    
    # Calculate and print total execution time
    end_time = time.time()
    total_time = end_time - script_start_time
    mins = total_time / 60
    hours = mins / 60
    if mins < 60:
        total_str = f"{mins:.2f} min"
    else:
        total_str = f"{hours:.2f} hr ({mins:.2f} min)"
    step_times["Total execution time"] = total_str
    
    # Get final memory usage
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_memory_mb = usage.ru_maxrss / 1024
    step_times["Peak memory usage"] = f"{peak_memory_mb:.2f} MB"
    
    # Add data summary
    step_times["Total data rows"] = f"{len(combined_df):,}"
    step_times["Along-track rows"] = f"{len(along_track_data_df):,}"
    if (plus_smap or plus_sic) and len(smap_data_df) > 0:
        step_times["SMAP/SIC rows"] = f"{len(smap_data_df):,}"
    
    # Write timing log to file
    timing_log_path = os.path.join(save_dir, "timing_log.txt")
    with open(timing_log_path, "w") as f:
        for step, t in step_times.items():
            f.write(f"{step}: {t}\n")
    
    print("\n=== Timing summary ===")
    for step, t in step_times.items():
        print(f"{step}: {t}")
    print(f"\nTiming log saved to: {timing_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_days", type=str, required=True)
    parser.add_argument("--smap", type=str, required=True)
    parser.add_argument("--date", type=str, required=False, help="Date for results directory in YYYYMMDD format (default: today)")
    parser.add_argument("--target_date", type=str, required=True, help="Target date for analysis in YYYY-MM-DD format")
    parser.add_argument("--smoothed_params_file_path", type=str, default=None, help="Absolute path to an H5 containing *_SMOOTHED tables. If provided, the run will load parameters from this file (artifacts will be auto-copied into the new results directory). If omitted, the run will optimise and smooth parameters.")
    parser.add_argument("--output_results_path", type=str, default=None, help="Full path to base results directory (e.g., /explore/nobackup/people/aapetty/GPSat_results). Default: ./results")
    parser.add_argument("--dataset_version", type=str, default=None, help="Dataset version string for organizing outputs (e.g., v1.0, production_v1). Outputs will be saved to {output_results_path}/{dataset_version}/{variable}/...")
    parser.add_argument("--variable", type=str, default="thickness", help="Variable name: thickness, total_freeboard, snow_depth, etc. Outputs saved to {output_results_path}/{dataset_version}/{variable}/...")
    parser.add_argument("--sic", type=str, default="false", help="Use SIC as ice-edge guide (0 where SIC < 0.15). Use with --smap false for freeboard/snow_depth. Default: false")
    parser.add_argument("--sic_use_prediction_day_only", type=str, default="true", help="Only load SIC for the prediction target date (default: true). Set false to load SIC for all days in window.")
    parser.add_argument("--smap_use_prediction_day_only", action='store_true', default=False, help="Only load SMAP data for the prediction day (faster, but less SMAP data). Default: False (load SMAP for all days in range)")
    parser.add_argument("--is2_pref", type=str, default="false", help="Prefer IS-2 over SMAP/SIC: drop SMAP/SIC points within 10 km of any IS-2 on same day; SIT_input per day use IS-2 where valid else SMAP/SIC. Default: false")
    args = parser.parse_args()

    print(args.num_days)
    print("smap:", args.smap, "sic:", args.sic, "is2_pref:", args.is2_pref)
    plus_smap = str2bool(args.smap)
    plus_sic = str2bool(args.sic)
    is2_pref = str2bool(args.is2_pref)
    sic_use_prediction_day_only = str2bool(args.sic_use_prediction_day_only)
    
    main(args.num_days,
        plus_smap=plus_smap,
        plus_sic=plus_sic,
        target_date=args.target_date,
        output_results_path=args.output_results_path,
        smoothed_params_file_path=args.smoothed_params_file_path,
        dataset_version=args.dataset_version,
        variable=args.variable,
        smap_use_prediction_day_only=args.smap_use_prediction_day_only,
        sic_use_prediction_day_only=sic_use_prediction_day_only,
        is2_pref=is2_pref)