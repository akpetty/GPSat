import os
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = info, 2 = warning, 3 = error
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


# Suppress HDF5 warnings
import h5py
h5py._errors.silence_errors()
# Import local modules
from extra_funcs import read_IS2SITMOGR4, along_track_preprocess, bin_to_IS2, cdr_preprocess_nh, load_sic_data_for_date

# Configure TensorFlow to handle GPU/CPU gracefully
import tensorflow as tf

# Fix Keras compatibility
try:
    import keras
    # Force use of legacy Keras if needed
    if not hasattr(keras, '__internal__'):
        os.environ['TF_USE_LEGACY_KERAS'] = 'True'
        import tensorflow.keras as keras
except ImportError:
    pass

# Ensure griddata is properly imported
from scipy.interpolate import griddata

physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU devices found. Running on CPU.")
else:
    print(f"Found {len(physical_devices)} GPU devices. Using GPU.")
    # Optional: Configure memory growth
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


import GPSat
from GPSat import get_data_path, get_parent_path
from GPSat.dataprepper import DataPrep
from GPSat.dataloader import DataLoader
from GPSat.utils import stats_on_vals, WGS84toEASE2, EASE2toWGS84, cprint, grid_2d_flatten, get_weighted_values, dataframe_to_2d_array
from GPSat.local_experts import LocalExpertOI, get_results_from_h5file
from GPSat.plot_utils import plot_wrapper, plot_pcolormesh, get_projection, plot_pcolormesh_from_results_data, plot_hyper_parameters
from GPSat.postprocessing import smooth_hyperparameters

print('loaded envs')


# Suppress logging
err = StringIO()
sys.stderr = err
for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

def load_data_around_target_date(config=None, IS2=None, plus_sic=False):
    """
    Load data for N days around a target date.
    SIC loading uses load_sic_data_for_date from extra_funcs (same as IS2_SMAP_GPSat_train).
    """
    target_date = config['target_date']
    num_days_before_after = config['num_days_before_after']
    beam = config['beam']
    
    along_track_data_all = []
    sic_data_all = []
    sic_data_gridded_all = []
    along_track_data_gridded_all = []
    
    # Parse target date
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Calculate date range
    start_dt = target_dt - timedelta(days=num_days_before_after)
    end_dt = target_dt + timedelta(days=num_days_before_after)
    
    # Process each day in range
    current_dt = start_dt
    while current_dt <= end_dt:
        year_str = str(current_dt.year)
        month_str = f"{current_dt.month:02d}"
        day_str = f"{current_dt.day:02d}"
        current_date_str = f"{year_str}-{month_str}-{day_str}"
        
        print(f"Date of data loading: {current_date_str}, beam: {beam}")
        
        # Load SIC for this date when plus_sic (same path/pattern as IS2_SMAP_GPSat_train via extra_funcs)
        if plus_sic:
            try:
                sic_data, sic_data_gridded = load_sic_data_for_date(current_date_str, IS2, config)
                if len(sic_data) > 0:
                    sic_data_all.append(sic_data)
                    sic_data_gridded_all.append(sic_data_gridded)
            except Exception as e:
                print(f'Error loading SIC for {current_date_str}: {e}')
        
        # Construct file pattern for along-track
        file_pattern = f'/explore/nobackup/people/aapetty/IS2thickness/rel006/run_adapt_4/final_data_along_track/002/{year_str}-{month_str}/data/sm/IS2SITDAT4_01_{year_str}{month_str}{day_str}*{beam}gt1l_002_sm.nc'
        file_list = glob.glob(file_pattern)
        
        for file in file_list:
            print('Reading file:', file)
            try:
                along_track_data = xr.open_mfdataset(file, engine='h5netcdf', preprocess=along_track_preprocess, combine='nested')
                
                # Subsample based on N_subsample to reduce input data size
                along_track_data = along_track_data.isel(along_track_distance_section=slice(None, None, config['N_subsample']))  
                along_track_data_all.append(along_track_data)
                along_track_data_gridded_all.append(bin_to_IS2(along_track_data, IS2))
                along_track_data.close()
            except KeyError as e:
                print(f"Error processing file {file}: {e}")
                continue
        
        current_dt += timedelta(days=1)
    
    # Concatenate the data
    along_track_data_all = xr.concat(along_track_data_all, 'along_track_distance_section')
    along_track_data_gridded_all = xr.concat(along_track_data_gridded_all, 'time')
    
    return along_track_data_all, along_track_data_gridded_all, sic_data_all, sic_data_gridded_all

def str2bool(s):
        return s.lower() in ['true', '1', 'yes', 'y']

def plot_thickness_comparison(dataset, target_date, figsize=(15, 5), extra_var='', out_str='', days_str='', save_dir='./results/'):
    """
    Create a comparison plot of two thickness datasets with their difference.
    """
    array1 = dataset.SIT.isel(time=0)
    array2 = dataset.SIT_input
    lon = dataset.lon
    lat = dataset.lat
    
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
    
    # Plot thickness maps
    thickness_plots = []
    for ax, data, title in zip(axs, [array1, array2], 
                              [f'(a) GPSat {extra_var} ({target_date})',
                               f'(b) Binned input data +/- '+days_str+' days']):
        p = ax.pcolormesh(lon, lat, data,
                         vmin=0., vmax=5,
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
                                  vmin=-2, vmax=2,
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
                label='Thickness [m]',
                orientation='horizontal',
                extend='max')
    fig.colorbar(diff_plot, cax=cbar_ax2,
                label='Thickness [m; difference]',
                orientation='horizontal',
                extend='both')
    
    plt.subplots_adjust(bottom=0.07, wspace=0.02)

    # Example usage:

    plt.savefig(save_dir+'/GPSat_'+out_str+'.png', dpi=300, facecolor="white", bbox_inches='tight')
    return


def plot_thickness_std(array, target_date, figsize=(9.6, 8), out_str = '', save_dir='./results/'):
    """
    Create a plot of thickness standard deviation.
    
    Parameters:
    -----------
    array : xarray.DataArray
        Thickness standard deviation dataset (SIT_var)
    target_date : str
        Date string for plot title
    figsize : tuple, optional
        Figure size in inches (width, height)
    """
    # Create figure and subplot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(-45, 90))
    
    # Plot settings
    extent = [-180, 180, 55, 90]
    
    # Plot standard deviation
    p = ax.pcolormesh(array.lon, array.lat, array,
                     vmin=0., vmax=3.,  # Adjust these limits based on your data
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
    ax.set_title(f'Standard deviation of GPSat thickness')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.04])  # [left, bottom, width, height]
    fig.colorbar(p, cax=cbar_ax, 
                label='Standard Deviation [m]',
                orientation='horizontal',
                extend='max')
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.1)
    
    # Uncomment to save figure
    plt.savefig(save_dir+'/GPSat_thickness_std_'+out_str+'.png', dpi=300, facecolor="white", bbox_inches='tight')
    return



def main(num_days, plus_sic = False, test_name = 'test4', config_overrides=None):
    
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
    if plus_sic:
        out_str = test_name+'_sic_'+num_days+'days'
    else:
        out_str = test_name+'_'+num_days+'days'
    print(out_str)
    
    config = {
        'target_date': '2019-02-15',
        'num_days_before_after': int(num_days),
        'beam': 'bnum1',
        'sic_cutoff': 0.15,
        'sic_coarsen_factor': 2,
        'sic_base_path': '/panfs/ccds02/home/aapetty/nobackup_symlink/Data/ICECONC/CDR/daily/final/v6',
        'sic_use_s3_fallback': False,
        'N_subsample': 2,
        'noise_std': 0.3,
        'expert_spacing': 200_000,  # balance: fine enough for Fram Strait, not overkill for central Arctic
        'training_radius': 400_000, # enough data for central Arctic; not too wide for narrow strait
        'inference_radius': 500_000, # allow 500–1000 km scales in central Arctic; GP learns shorter in strait
        'model_type': 'GPflowSGPRModel', #oi_modes = ['GPflowGPRModel', 'GPflowSGPRModel', 'GPflowSVGPModel', 'sklearnGPRModel', 'GPflowVFFModel', 'GPflowASVGPModel']
        'pred_spacing': 25_000,
        'val_col': 'ice_thickness',
        'max_iter': 10000,
        'out_path': out_str,
        'use_region_filter': True,
        'filter_expert_locations': True,
        'filter_prediction_locations': True,
        'allowed_regions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #Outside_of_defined_regions Central_Arctic Beaufort_Sea Chukchi_Sea East_Siberian_Sea Laptev_Sea Kara_Sea Barents_Sea    East_Greenland_Sea Baffin_Bay_and_Davis_Strait Gulf_of_St._Lawrence Hudson_Bay Canadian_Archipelago Bering_Sea    Sea_of_Okhotsk Sea_of_Japan Bohai_Sea Gulf_of_Bothnia Baltic_Sea Gulf_of_Alaska Land Coast Lakes";
    }

    log_step("Configuration", step_start, time.time())

    
    # Create directory if it doesn't exist
    step_start = time.time()
    print("2. Creating output directory...")
    save_dir = './results/'+config['out_path']
    os.makedirs(save_dir, exist_ok=True)
    # Save config to CSV
    config_df = pd.DataFrame(list(config.items()), columns=['Parameter', 'Value'])
    config_df.to_csv(save_dir+'/config.csv', index=False)
    log_step("Create output directory & save config", step_start, time.time())
    
    ### Initialize S3 filesystem
    step_start = time.time()
    print("3. Initializing S3 filesystem...")
    s3 = s3fs.S3FileSystem(anon=True)
    log_step("Initialize S3 filesystem", step_start, time.time())
    
    ### Read in the monthly gridded IS2 data
    step_start = time.time()
    print("4. Reading IS2 data...")
    IS2 = read_IS2SITMOGR4(data_type='netcdf-local', version='003', 
                           local_data_path="/explore/nobackup/people/aapetty/IS2thickness/rel006/run_adapt_4/final_data_gridded/").rename({'longitude':'lon','latitude':'lat'})
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
    along_track_data, along_track_data_gridded, sic_data, sic_data_gridded = load_data_around_target_date(
        config=config,
        IS2=IS2,
        plus_sic=plus_sic
    )
    log_step("Load data around target date", step_start, time.time())

    
    ### Process data
    step_start = time.time()
    print("8. Processing data...")
    if plus_sic and sic_data:
        sic_data_df = pd.concat(sic_data).dropna().reset_index()
        sic_data_df['time'] = sic_data_df.time.values.astype("datetime64[D]").astype(float)
        print('SIC times:', sic_data_df.time.values)
        sic_data_df = sic_data_df.astype('float64')
        if 'index' in sic_data_df.columns:
            sic_data_df = sic_data_df.drop(columns=['index'])
        # load_sic_data_for_date returns column config['val_col'] (ice_thickness for this script)
    else:
        sic_data_df = pd.DataFrame()

    along_track_data_df = along_track_data.to_dataframe().dropna().reset_index()
    along_track_data_df['time'] = along_track_data_df.time.values.astype("datetime64[D]").astype(float)
    along_track_data_df = along_track_data_df.astype('float64')
    along_track_data_df = along_track_data_df.drop(columns=['along_track_distance_section', 'lat', 'lon'])
    log_step("Process data", step_start, time.time())

    
    ### Combine dataframes
    step_start = time.time()
    print("9. Combining dataframes...")
    if plus_sic:
        combined_df = pd.concat([along_track_data_df, sic_data_df], ignore_index=True)
        combined_df = combined_df.sort_values(by='time')
    else:
        combined_df = along_track_data_df
    log_step("Combine dataframes", step_start, time.time())

    
    ### Set up data configuration
    step_start = time.time()
    print("10. Setting up data configuration...")
    data = {
        "data_source": combined_df, 
        "obs_col": "ice_thickness",
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
    print("11. Setting up local expert configuration...")
    local_expert = {
        "source": eloc
    }
    log_step("Set up local expert configuration", step_start, time.time())
    
    ### Set up model configuration
    step_start = time.time()
    print("12. Setting up model configuration...")
    
    model = {
        "oi_model": config['model_type'],
        "init_params": {
            "likelihood_variance": config['noise_std']**2,
            "coords_scale": [25000, 25000, 1],
            "jitter": 1e-6  # Add jitter for numerical stability?
        },
        #"load_params":{
        #    "file": "/explore/nobackup/people/aapetty/GitHub/GPSat/results/test500k_sic_15days/IS2_interp_test_petty_v1.h5",
        #    "param_names": ["likelihood_variance","kernel_variance","lengthscales"]
        #},
        "optim_kwargs": {
            #"fixed_params": ['likelihood_variance'],#FIXING THE LIKLIHOOD FUNCTION TO INCREASE SPEED?
            "max_iter": config['max_iter']
        },
        "constraints": {
            "lengthscales": {
                "low": [10000, 10000, 0.5], 
                "high": [1_000_000, 1_000_000, 50]  # allow ~500–1000 km in central Arctic
            },
            "likelihood_variance": {
                "low": 1e-3, 
                "high": 0.5
            },
            "kernel_variance": {
                "low": 1e-3, 
                "high": 5
            }
        }
    }
    log_step("Set up model configuration", step_start, time.time())
    
    ### Set up prediction locations
    step_start = time.time()
    print("13. Setting up prediction locations...")
    pred_loc = {
        "method": "from_dataframe",
        "df": ploc,
        "max_dist": config['inference_radius']
    }
    log_step("Set up prediction locations", step_start, time.time())

    ### Initializing LocalExpertOI
    step_start = time.time()
    print("14. Initializing LocalExpertOI...")
    store_path = get_parent_path("results/"+config['out_path'], "IS2_interp_test_petty_v1.h5")
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
    print("14.2. Running LocalExpertOI...")
    with HiddenPrints():
        try:
            locexp.run(store_path=store_path,
                        optimise=True,
                        check_config_compatible=False)
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
    print("15. Extracting results...")
    try:
        dfs, oi_config = get_results_from_h5file(store_path)
        print(f"tables in results file: {list(dfs.keys())}")
    except Exception as e:
        print(f"Error extracting results: {str(e)}")
        raise
    log_step("Extract results (unsmoothed)", step_start, time.time())

    ### Plot and save results
    step_start = time.time()
    print("16. Plotting and saving results...")
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
    print("17. Smoothing hyperparameters...")
    try:
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
        # length scales for the kernel smoother in each dimension
        # - as well as any min/max values to apply
        "smooth_config_dict": {
            "lengthscales": {
                "l_x": 200_000,
                "l_y": 200_000
            },
            #"likelihood_variance": {
            #    "l_x": 200_000,
            #    "l_y": 200_000,
            #    "max": 0.3    ####cannot set to 0, so set to something close to 0 (if confident in observations or want to validate method) WAS 0.17
            #},
            "kernel_variance": {
                "l_x": 200_000,
                "l_y": 200_000,
                "max": 3     ####not sure if this should be set, .1 by default (if too high, can lead to overfitting) WAS 2.34
            }
        },
        "save_config_file": True
        }
        
        smooth_result_config_file = smooth_hyperparameters(**smooth_config)
        print(f"Smooth result config file: {smooth_result_config_file}")
        # modify the model configuration to include "load_params"
        # DONT REALLY GET THIS
        model_load_params = model.copy()
        model_load_params["load_params"] = {
            "file": store_path,
            "table_suffix": smooth_config["table_suffix"]
        }
    except Exception as e:
        print(f"Error smoothing hyperparameters: {str(e)}")
        raise
    log_step("Smooth hyperparameters", step_start, time.time())

    ### Run LocalExpertOI with smoothed hyperparameters
    step_start = time.time()
    print("17.1. Running LocalExpertOI with smoothed hyperparameters")
    try:    
        locexp_smooth = LocalExpertOI(expert_loc_config=local_expert,
                            data_config=data,
                            model_config=model_load_params,
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
        print("17.1. LocalExpertOI run completed successfully")
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
    print("18. Plotting smoothed hyperparameters...")
    try:
        # plot and save results
        fig = plot_hyper_parameters(dfs,
                        coords_col=oi_config[0]['data']['coords_col'],  # ['x', 'y', 't']
                        row_select=None,
                        table_names=["lengthscales", "kernel_variance", "likelihood_variance"],
                        #table_suffix=smooth_config["table_suffix"],
                        table_suffix = '_SMOOTHED',
                        plot_template=plot_template,
                        plots_per_row=3,
                        suptitle="smoothed hyper params",
                        qvmin=0.01,
                        qvmax=0.99)
        plt.tight_layout()
        plt.savefig(save_dir+'/GPSat_smoothed_weights_'+config['target_date']+'.png', dpi=300, facecolor="white", bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error plotting smoothed hyperparameters: {str(e)}")
        raise
    log_step("Plot and save smoothed hyperparameters", step_start, time.time())
    

    ### Generate netcdf file
    step_start = time.time()
    print('19. Generating netcdf file...')
    try:
        # plot and save results
        smoothed=True
        if smoothed:
            plt_data = dfs["preds"+"_SMOOTHED"]
            out_path_pred = "IS2_interp_test_petty_"+config['target_date']+".nc"
            extra_var=''
        else:
            plt_data = dfs["preds"]
            out_path_pred = "IS2_interp_test_petty_"+config['target_date']+"_unsmoothed.nc"
            extra_var='unsmoothed'
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
        
        # ADD THE BINNED DATA TO THE DATASET
        along_track_data_gridded = along_track_data_gridded.sortby('time')
        # Resample the binned along-track input data to daily
        along_track_data_gridded_daily = along_track_data_gridded.resample(time='1D').mean().assign_coords({'lat':IS2.lat,'lon':IS2.lon})
        array2 = along_track_data_gridded_daily.ice_thickness.mean(dim='time', skipna=True)
        if plus_sic and sic_data_gridded:
            # Concatenate all datasets along the time dimension
            sic_data_gridded_daily = xr.concat(sic_data_gridded, dim='time')
            sic_data_gridded_daily = sic_data_gridded_daily.assign_coords({'lat':IS2.lat,'lon':IS2.lon})
            array3 = sic_data_gridded_daily.ice_thickness.mean(dim='time', skipna=True)
            combined = xr.concat([array2, array3], dim='time')
        else:
            combined = array2
        
        # Take the nanmean along the time dimension
        mean_input_data = xr.DataArray(
            np.nanmean(combined.values, axis=0) if plus_sic else combined.values,  # axis=0 for time dimension
            dims=combined.dims[1:] if plus_sic else combined.dims,  # exclude time dimension
            coords={k: v for k, v in combined.coords.items() if k != 'time'} if plus_sic else combined.coords
        )
        IS2_interp['SIT_input'] = mean_input_data
        print(IS2_interp)
        

        store_path_nc = get_parent_path("results/"+config['out_path'], out_path_pred)
        IS2_interp.to_netcdf(store_path_nc)
    except Exception as e:
        print(f"Error generating netcdf file: {str(e)}")
        raise
    log_step("Generate netcdf file", step_start, time.time())
        
    ### Plot SIT and SIT_VAR
    step_start = time.time()
    print("20. Plotting SIT and SIT_VAR...")

    try:
        plot_thickness_comparison(IS2_interp, target_date='2019-04-15', out_str=extra_var+config['target_date'], days_str = str(config['num_days_before_after']), save_dir=save_dir)
        # Plot standard deviation
        plot_thickness_std(IS2_interp.SIT_var[0], target_date='2019-04-15', out_str=extra_var+config['target_date'], save_dir=save_dir)
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
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--sic", type=str, required=True)
    args = parser.parse_args()

    print(args.num_days)
    print(args.name)
    print(str(args.sic))
    plus_sic = str2bool(args.sic)
    main(args.num_days, plus_sic=plus_sic, test_name=args.name)