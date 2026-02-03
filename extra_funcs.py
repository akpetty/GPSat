# New functions used in the all-season notebooks

""" extra_funcs.py 

Copied from the all-season notebooks
"""

# Regular Python library imports 
import xarray as xr 
import numpy as np
import pandas as pd
import pyproj
import scipy.interpolate
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Interpolating/smoothing packages 
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel

from GPSat.utils import stats_on_vals,  WGS84toEASE2, EASE2toWGS84, cprint, grid_2d_flatten, get_weighted_values, dataframe_to_2d_array
from GPSat.dataprepper import DataPrep

def bin_to_IS2(ds, IS2, rs='1D', val_col='ice_thickness'):
    # AP: Added IS-2. val_col: column name for the value to bin (ice_thickness, total_freeboard, snow_depth, etc.)
    
    # Check if input is already a DataFrame
    if isinstance(ds, pd.DataFrame):
        df = ds
    else:
        df = ds.to_dataframe().dropna().reset_index()
    
    if val_col not in df.columns:
        raise ValueError(f"bin_to_IS2: column '{val_col}' not in data (columns: {list(df.columns)})")
    #convenience function from GPsat to bin the along track data onto the ICESat-2 grid
    bin_ds = DataPrep.bin_data_by(df=df,
                              by_cols=['time'],
                              val_col=val_col,
                              x_col='x',
                              y_col='y',
                              grid_res=25_000,
                              limit = 200_000,
                              x_range=[IS2.x.min()-(25_000/2), IS2.x.max()+(25_000/2)],
                              y_range=[IS2.y.min()-(25_000/2), IS2.y.max()+(25_000/2)])
    return bin_ds.resample(time=rs).mean()

###old function for removing a fraction of files on a given day for training 
###the new method removes a region for training after processing data
def split_files_by_day(file_list, target_day=15, percentage=0.7):
    # Filter files that correspond to the target day
    day_files = [file for file in file_list if f'201904{target_day}' in file]
    # Calculate how many files to remove
    num_files_to_remove = int(len(day_files) * percentage)
    # Randomly select files to remove from the matching day files
    files_to_remove = random.sample(day_files, num_files_to_remove)
    # Create a new list for remaining files (either from other days or files from the target day that weren't removed)
    remaining_files = [file for file in file_list if file not in files_to_remove]
    return files_to_remove, remaining_files

def along_track_preprocess(ds, data_variable='ice_thickness'):
    """
    Preprocess along-track data for training.
    
    Args:
        ds (xarray.Dataset): Input dataset
        data_variable (str): Variable to extract (ice_thickness, total_freeboard, snow_depth, etc.). Default ice_thickness.
    """
    lat_0 = 90
    lon_0 = -45
    ds = ds.set_coords(["longitude", "latitude","gps_seconds"]).rename({'latitude':'lat','longitude':'lon','gps_seconds':'time'})
    if data_variable in ds.data_vars:
        ds = ds[data_variable]
    elif data_variable in ds and hasattr(ds, data_variable):
        ds = getattr(ds, data_variable)
    else:
        ds = ds.ice_thickness  # fallback for thickness-only datasets
    #ds['time'] = ds.time + pd.to_datetime('1980-01-06T00:00:00.000000')
    ref_time = np.datetime64("1980-01-06T00:00:00")
    ds['time'] = ref_time + ds['time'].astype("timedelta64[s]")
    #convenience function from GPsat to get x, y grid coordinates from the WGS84 lat/lon grid 
    x_grid,y_grid = WGS84toEASE2(lon=ds['lon'], lat=ds['lat'], lat_0=lat_0, lon_0=lon_0)
    ds['x'], ds['y'] = (xr.DataArray(x_grid,coords={'along_track_distance_section':ds.along_track_distance_section}),
                                            xr.DataArray(y_grid,coords={'along_track_distance_section':ds.along_track_distance_section}))
    return ds
 

def cdr_preprocess_nh(ds, IS2):
    """
    Preprocess CDR data and align with IS2 coordinates.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input CDR dataset
    IS2 : xarray.Dataset
        IS2 dataset to align coordinates with
        
    Returns:
    --------
    xarray.DataArray
        Preprocessed CDR data aligned with IS2 coordinates
    """
    # Rename coordinates to match IS2
    ds = ds.rename({'xgrid':'x','ygrid':'y','tdim':'time'})
    ds = ds.set_coords(["y", "x"])
    
    # Align coordinates with IS2
    ds = ds.assign_coords(
        lat=IS2.lat,
        lon=IS2.lon,
        time=ds.time.compute()
    )
    
    # Return the appropriate variable based on what's available
    if 'cdr_seaice_conc_monthly' in ds.data_vars:
        return ds.cdr_seaice_conc_monthly
    else: 
        return ds.cdr_seaice_conc
    
def read_IS2SITMOGR4(data_type='zarr-s3', version='V3', local_data_path="./data/IS2SITMOGR4/", 
                     zarr_path='s3://icesat-2-sea-ice-us-west-2/IS2SITMOGR4_V3/IS2SITMOGR4_V3_201811-202404.zarr',
                     netcdf_s3_path='s3://icesat-2-sea-ice-us-west-2/IS2SITMOGR4_V3/netcdf/', 
                     persist=True): 
    """ Read in IS2SITMOGR4 monthly gridded thickness dataset from local netcdf files, 
    download the netcdf files from S3 storage, or read in the aggregated zarr dataset from S3. 
    Currently supports either Version 2 (V2) or Version 3 (V3) data. 
    
    Args: 
        data_type (str, required): (default to "zarr-s3", but also "netcdf-s3" or "netcdf-local" which is a local version of the netcdf files)
        version (str, required): dataset version, the default is V3 but V2 has some little changes we need to adapt for.
        local_data_path (str, required): local data directory
        zarr_path (str): path to zarr file
        netcdf_s3_path (str): path to netcdf files stored on s3
        persist (boleen): if zarr option decide if you want to persist (load) data into memory

    Returns: 
        is2_ds (xr.Dataset): aggregated IS2SITMOGR4 xarray dataset, dask chunked/virtually allocated in the case of the zarr option (or allocated to memory if persisted). 
        
    Version History: 
        November 2025:
         - Updated to read V4 data
         - That includes the time coordinate now
        February 2025
            - hard-coded the datapaths as mostly just V3 at this point and the V2/V3 stuff was getting confusing
            - now you just provide the path to the zarr or netcdf files as desired which I think is easier. 
     
        November 2023 (for V3 data release):  
            - Moved the download code to it's own section at the start of the function
            - Changed local paths
            - Baked in the date_str label as that is just a function of the dataset version anyway
            - Adapted the netcdf reader to use open_mfdataset, required a preprocessing data dimension step. Much more elegant!
            Note than in Version 3 there was a change in the xgrid/ygrid coordinates to x/y.
    """
            
    if data_type=='zarr-s3':

        print('load zarr from S3 bucket')

        print('zarr_path:', zarr_path)
        s3 = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=zarr_path, s3=s3, check=False)
        is2_ds = xr.open_zarr(store=store)
        #print(is2_ds)
        # Had a problem with these being loaded as dask arrays which cartopy doesnt love
        is2_ds = is2_ds.assign_coords(longitude=(["y","x"], is2_ds.longitude.values))
        is2_ds = is2_ds.assign_coords(latitude=(["y","x"], is2_ds.latitude.values))

        if persist==True:
            is2_ds = is2_ds.persist()
        
        return is2_ds

    if data_type=='netcdf-s3':
        # Download data from S3 to local bucket
        print("download from S3 bucket: ", netcdf_s3_path)

        # Download netCDF data files
        fs = s3fs.S3FileSystem(anon=True)

        #files references the entire bucket.
        files = fs.ls(netcdf_s3_path)
        for file in files:
            print('Downloading file from bucket to local storage...', file)
            fs.download(file, local_data_path+version+'/')

    # Read in files for each month as a single xr.Dataset
    print('Searching for files in: ', local_data_path+version+'/*.nc')
    filenames = glob.glob(local_data_path+version+'/*.nc')
    if len(filenames) == 0: 
        raise ValueError("No files, exit")
        return None
    
    # Add a dummy time then add the dates I want, seemed the easiest solution
    if version=='V2':
        is2_ds = xr.open_mfdataset(filenames, preprocess = add_time_dim_v2, engine='netcdf4')
    elif version=='V3':
        is2_ds = xr.open_mfdataset(filenames, preprocess = add_time_dim_v3, engine='netcdf4')
    else:
        is2_ds = xr.open_mfdataset(filenames, engine='netcdf4')
    
    #dates = [pd.to_datetime(file.split("IS2SITMOGR4_01_")[1].split("_")[0], format = "%Y%m")  for file in filenames]
    #is2_ds["time"] = dates

    # Sort by time as glob file list wasn't!
    is2_ds = is2_ds.sortby("time")
    if version=='V2':
        is2_ds = is2_ds.set_coords(["latitude","longitude","xgrid","ygrid"]) 
    else:
        is2_ds = is2_ds.set_coords(["latitude","longitude","x","y"])
    
    # Drop time dimension from longitude and latitude if they have it
    # These are static variables that shouldn't have a time dimension
    if 'time' in is2_ds.longitude.dims:
        is2_ds['longitude'] = is2_ds['longitude'].isel(time=0, drop=True)
    if 'time' in is2_ds.latitude.dims:
        is2_ds['latitude'] = is2_ds['latitude'].isel(time=0, drop=True)
    
    is2_ds = is2_ds.assign_coords(longitude=(["y","x"], is2_ds.longitude.values))
    is2_ds = is2_ds.assign_coords(latitude=(["y","x"], is2_ds.latitude.values))
    
    is2_ds = is2_ds.assign_attrs(description="Aggregated IS2SITMOGR4 "+version+" dataset.")
    print(is2_ds.head)
    return is2_ds

def get_summer_data(da, year_start=None, start_month="May", end_month="Jul", force_complete_season=False):
    """ Select data for summer seasons corresponding to the input time range 
    
    Args: 
        da (xr.Dataset or xr.DataArray): data to restrict by time; must contain "time" as a coordinate 
        year_start (str, optional): year to start time range; if you want Sep 2019 - Apr 2020, set year="2019" (default to the first year in the dataset)
        start_month (str, optional): first month in winter (default to September)
        end_month (str, optional): second month in winter; this is the following calender year after start_month (default to April)
        force_complete_season (bool, optional): require that winter season returns data if and only if all months have data? i.e. if Sep and Oct have no data, return nothing even if Nov-Apr have data? (default to False) 
        
    Returns: 
        da_summer (xr.Dataset or xr.DataArray): da restricted to winter seasons 
    
    """
    if year_start is None: 
        print("No start year specified. Getting winter data for first year in the dataset")
        year_start = str(pd.to_datetime(da.time.values[0]).year)
    
    start_timestep = start_month+" "+str(year_start) # mon year 
    end_timestep = end_month+" "+str(year_start) # mon year
    summer = pd.date_range(start=start_timestep, end=end_timestep, freq="MS") # pandas date range defining winter season
    months_in_da = [mon for mon in summer if mon in da.time.values] # Just grab months if they correspond to a time coordinate in da

    if len(months_in_da) > 0: 
        if (force_complete_season == True) and (all([mon in da.time.values for mon in summer])==False): 
            da_summer = None
        else: 
            da_summer = da.sel(time=months_in_da)
    else: 
        da_summer = None
        
    return da_summer


def compute_gridcell_summer_means(da, years=None, start_month="May", end_month="Jul", force_complete_season=False): 
    """ Compute summer means over the time dimension. Useful for plotting as the grid is maintained. 
    
    Args: 
        da (xr.Dataset or xr.DataArray): data to restrict by time; must contain "time" as a coordinate 
        years (list of str): years over which to compute mean (default to unique years in the dataset)
        year_start (str, optional): year to start time range; if you want Nov 2019 - Apr 2020, set year="2019" (default to the first year in the dataset)
        start_month (str, optional): first month in winter (default to November)
        end_month (str, optional): second month in winter; this is the following calender year after start_month (default to April)
        force_complete_season (bool, optional): require that winter season returns data if and only if all months have data? i.e. if Sep and Oct have no data, return nothing even if Nov-Apr have data? (default to False) 
    
    Returns: 
        merged (xr.DataArray): DataArray with summer means as a time coordinate
    """
    
    if years is None: 
        years = np.unique(pd.to_datetime(da.time.values).strftime("%Y")) # Unique years in the dataset 

    summer_means = []
    for year in years: # Loop through each year and grab the summer months, compute winter mean, and append to list 
        da_summer_i = get_summer_data(da, year_start=year, start_month=start_month, end_month=end_month, force_complete_season=force_complete_season)
        if da_summer_i is None: 
            continue
        da_mean_i = da_summer_i.mean(dim="time", keep_attrs=True) # Compute mean over time dimension

        # Assign time coordinate 
        time_arr = pd.to_datetime(da_summer_i.time.values)
        da_mean_i = da_mean_i.assign_coords({"time":time_arr[0].strftime("%b %Y")+" - "+time_arr[-1].strftime("%b %Y")})
        da_mean_i = da_mean_i.expand_dims("time")

        summer_means.append(da_mean_i)

    merged = xr.merge(summer_means) # Combine each summer mean Dataset into a single Dataset, with the time period maintained as a coordinate
    merged = merged[list(merged.data_vars)[0]] # Convert to DataArray
    merged.time.attrs["description"] = "Time period over which mean was computed" # Add descriptive attribute 
    return merged 



def add_time_dim_v3(xda):
    """ dummy function to just set current time as a new dimension to concat files over, change later! """
    xda = xda.set_coords(["latitude","longitude", "x", "y"])
    xda = xda.expand_dims(time = [datetime.now()])
    return xda

def read_IS2SITMOGR4S(version='V0', local_data_path="./data/IS2SITMOGR4_SUMMER/"): 
    """ Read in IS2SITMOGR4 summer monthly gridded thickness dataset from local netcdf files

    """
    
    print(local_data_path+version+'/*.nc')
    filenames = glob.glob(local_data_path+version+'/*.nc')
    if len(filenames) == 0: 
        raise ValueError("No files, exit")
        return None
    
    dates = [pd.to_datetime(file.split("IS2SIT_SUMMER_01_")[1].split("_")[0], format = "%Y%m")  for file in filenames]
    # Add a dummy time then add the dates I want, seemed the easiest solution
    is2_ds = xr.open_mfdataset(filenames, preprocess = add_time_dim_v3, engine='netcdf4')
            
    is2_ds["time"] = dates

    # Sort by time as glob file list wasn't!
    is2_ds = is2_ds.sortby("time")
    is2_ds = is2_ds.set_coords(["latitude","longitude","x","y"])
    
    is2_ds = is2_ds.assign_coords(longitude=(["y","x"], is2_ds.longitude.values))
    is2_ds = is2_ds.assign_coords(latitude=(["y","x"], is2_ds.latitude.values))
    
    is2_ds = is2_ds.assign_attrs(description="Aggregated IS2SITMOGR4 summer "+version+" dataset.")

    return is2_ds


def getCS2ubris(mapProj, dataPathCS2, dataset):
    """ Read in the University of Bristol CryoSat-2 sea ice thickness data

    
    Args:
        dataPathCS2 (str): location of data
        
    Returns
        xptsT (2d numpy array): x coordinates on our map projection
        yptsT (2d numpy array): y coordinates on our map projection
        thicknessCS (2d numpy array): monthly sea ice thickness estimates
        

    """
    ubris_f = xr.open_dataset(dataPathCS2+dataset, decode_times=False)

    # Issue with time starting from year 0!
    # Re-set it to start from some other year
    ubris_f = ubris_f.rename({'Time':'time'})
    ubris_f['time'] = ubris_f['time']-679352
    ubris_f.time.attrs["units"] = "days since 1860-01-01"
    decoded_time = xr.decode_cf(ubris_f)

    ubris_f['time']=decoded_time.time
    ubris_f = ubris_f.swap_dims({'t': 'time'})

    # Resample to monthly, note that the S just makes the index start on the 1st of the month
    thicknessCS = ubris_f.resample(time="MS").mean()
    xptsT, yptsT = mapProj(thicknessCS.isel(time=0).Longitude, thicknessCS.isel(time=0).Latitude)
    
    return xptsT, yptsT, thicknessCS


def regridToICESat2(dataArrayNEW, xptsNEW, yptsNEW, xptsIS2, yptsIS2):  
    """ Regrid new data to ICESat-2 grid 
    
    Args: 
        dataArrayNEW (xarray DataArray): Numpy variable array to be gridded to ICESat-2 grid 
        xptsNEW (numpy array): x-values of dataArrayNEW projected to ICESat-2 map projection 
        yptsNEW (numpy array): y-values of dataArrayNEW projected to ICESat-2 map projection 
        xptsIS2 (numpy array): ICESat-2 longitude projected to ICESat-2 map projection
        yptsIS2 (numpy array): ICESat-2 latitude projected to ICESat-2 map projection
    
    Returns: 
        gridded (numpy array): data regridded to ICESat-2 map projection
    
    """
    #gridded = []
    #for i in range(len(dataArrayNEW.values)): 
    #gridded = scipy.interpolate.griddata((xptsNEW.flatten(),yptsNEW.flatten()), dataArrayNEW.flatten(), (xptsIS2, yptsIS2), method = 'nearest')
    try:
        #print('try method 1...')
        gridded = scipy.interpolate.griddata((xptsNEW.flatten(),yptsNEW.flatten()), dataArrayNEW.flatten(), (xptsIS2, yptsIS2), method = 'nearest')
    except:
        try:
            #print('Did not work, try method 2..')
            gridded = scipy.interpolate.griddata((xptsNEW,yptsNEW), dataArrayNEW, (xptsIS2, yptsIS2), method = 'nearest')
        except:
            print('Error interpolating..')
    
    return gridded


def regrid_ubris_to_is2(mapProj, xIS2, yIS2, out_lons, out_lats, date_range, dataPathCS2='/home/jovyan/Data/CS2/UIT/', dataset='ubristol_cryosat2_seaicethickness_nh_80km_v1p7.nc'):
    """
    Regrid UBRIS data to ICESat-2 grid

    Args:
        mapProj (Basemap): Basemap projection object
        xIS2 (numpy array): ICESat-2 x-coordinates
        yIS2 (numpy array): ICESat-2 y-coordinates
        out_lons (numpy array): Output longitudes
        out_lats (numpy array): Output latitudes
        date_range (list of str): List of dates to process
        dataPathCS2 (str, optional): Path to CryoSat-2 data (default is '/home/jovyan/Data/CS2/UIT/')
        dataset (str, optional): Dataset filename (default is 'ubristol_cryosat2_seaicethickness_nh_80km_v1p7.nc')

    Returns:
        cs2_ubris (xarray Dataset): Regridded UBRIS data on ICESat-2 grid
    """


    xptsIS2, yptsIS2 = np.meshgrid(xIS2, yIS2)


    cs2_ubris = []
    valid_dates=[]

    xptsT_ubris, yptsT_ubris, cs2_ubris_raw = getCS2ubris(mapProj, dataPathCS2, dataset)
    
    for date in date_range:
        #print(date)
        try:
            cs2_ubris_temp_is2grid = regridToICESat2(cs2_ubris_raw.Sea_Ice_Thickness.sel(time=date).values, xptsT_ubris, yptsT_ubris, xptsIS2, yptsIS2) 
            ice_conc_is2grid = regridToICESat2(cs2_ubris_raw.Sea_Ice_Concentration.sel(time=date).values, xptsT_ubris, yptsT_ubris, xptsIS2, yptsIS2)     
            #cs2_ubris_temp_is2grid[ice_conc_is2grid<0.5]=np.nan
            
            cs2_ice_type_is2grid = regridToICESat2(cs2_ubris_raw.Sea_Ice_Type.sel(time=date).values, xptsT_ubris, yptsT_ubris, xptsIS2, yptsIS2) 
            cs2_ice_density_is2grid = 917. - (cs2_ice_type_is2grid * (917. - 882.))

        except:
            print(date)
            print('no CS-2 data or issue with gridding, so skipping...')
            continue
        valid_dates.append(date)

        cs2_ubris_temp_is2grid_xr = xr.Dataset({'cs2_sea_ice_thickness_UBRIS': (('y', 'x'), cs2_ubris_temp_is2grid), 
                                'cs2_sea_ice_type_UBRIS': (('y', 'x'), cs2_ice_type_is2grid), 
                                'cs2_sea_ice_density_UBRIS': (('y', 'x'), cs2_ice_density_is2grid)}, 
                                coords = {'latitude': (('y','x'), out_lats), 'longitude': (('y','x'), out_lons), 'x': (('x'), xIS2),  'y': (('y'), yIS2)} 
                                )
        
        cs2_ubris.append(cs2_ubris_temp_is2grid_xr)

    cs2_ubris = xr.concat(cs2_ubris, 'time')
    #cs2_ubris = cs2_ubris.assign_coords(time=valid_dates)
    cs2_ubris_attrs = {'units': 'meters', 'long_name': 'University of Bristol CryoSat-2 Arctic sea ice thickness', 'data_download': 'https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01613', 
            'download_date': '09-2022', 'citation': 'Landy, J.C., Dawson, G.J., Tsamados, M. et al. A year-round satellite sea-ice thickness record from CryoSat-2. Nature 609, 517–522 (2022). https://doi.org/10.1038/s41586-022-05058-5'} 
    cs2_ubris = cs2_ubris.assign_coords(time=valid_dates)
    cs2_ubris = cs2_ubris.assign_attrs(cs2_ubris_attrs)  

    return cs2_ubris

def get_cs2is2_snow(mapProj, xIS2, yIS2, dataPathCS2='./data/uit_cs2-is2-ak_snow_depth_25km_v3.nc'):
    """
    Load, process and regrid CS2-IS2 snow depth data to the IS2 grid for winter months between 2018-2023.
    
    This function:
    1. Creates a time array for winter months (Oct-Apr) from 2018-2023
    2. Loads CS2-IS2 snow depth data from a netCDF file
    3. Regrids the data to match the IS2 grid coordinates
    4. Returns the regridded data as an xarray DataArray
    
    Parameters
    ----------
    mapProj : function
        Map projection function to convert lat/lon to x/y coordinates
    xIS2 : numpy.ndarray
        1D array of x-coordinates for the IS2 grid
    yIS2 : numpy.ndarray
        1D array of y-coordinates for the IS2 grid
    dataPathCS2 : str, optional
        Path to the CS2-IS2 snow depth netCDF file
        Default is './data/uit_cs2-is2-ak_snow_depth_25km_v3.nc'
    
    Returns
    -------
    xarray.DataArray
        Regridded snow depth data with dimensions:
        - time: winter months from 2018-2023
        - y: IS2 grid y-coordinates
        - x: IS2 grid x-coordinates
        
    Notes
    -----
    Winter season is defined as October through April of the following year.
    The data is regridded from the original CS2 grid to the IS2 grid using
    the regridToICESat2 function.
    """
    
    # Create date range for Oct-Apr periods from 2018-2023
    dates = []
    for year in range(2018, 2023):
        # October to December of current year
        dates.extend(pd.date_range(start=f'{year}-10-01', end=f'{year}-12-31', freq='MS'))
        # January to April of next year
        dates.extend(pd.date_range(start=f'{year+1}-01-01', end=f'{year+1}-04-30', freq='MS'))

    # Convert to numpy datetime64 array
    dates_cs2is2 = np.array(dates, dtype='datetime64[ns]')

    # Get the IS-2/CS-2 snow depths
    xptsIS2, yptsIS2 = np.meshgrid(xIS2, yIS2)
    

    cs2is2_snow = xr.open_dataset(dataPathCS2, decode_times=False)
    xptsT_ubris, yptsT_ubris = mapProj(cs2is2_snow.isel(t=0).Longitude, cs2is2_snow.isel(t=0).Latitude)

    cs2is2_snow_regridded = []
    for t in cs2is2_snow.t.values:
        #print(t)
        cs2is2_snow_is2grid = regridToICESat2(cs2is2_snow.sel(t=t).Snow_Depth_KuLa.values, 
                                                xptsT_ubris, yptsT_ubris, xptsIS2, yptsIS2) 
        cs2is2_snow_regridded.append(cs2is2_snow_is2grid)
    # Convert list to numpy array with proper dimensions
    cs2is2_snow_regridded_array = np.array(cs2is2_snow_regridded)

    # Create a new DataArray with the regridded data
    cs2is2_snow_regridded_da = xr.DataArray(
        cs2is2_snow_regridded_array,
        dims=['time', 'y', 'x'],
        coords={
            'time': dates_cs2is2,
            'y': yIS2,
            'x': xIS2
        },
        name='cs2is2_snow_depth'
    ) 
    cs2is2_snow_regridded_da  

    return cs2is2_snow_regridded_da

def apply_interpolation_time(dataset_og, xptsIS2, yptsIS2, variables, force_copy=False, method = "linear"):
    # Apply interpolation to all data for consistency
    # Interpolation settings 
    # Force copy is used to force a copy of the data to be made, otherwise the data will not be overwritten if already exists
    # Create a copy of the dataset to avoid modifying the original
    dataset = dataset_og.copy(deep=True)

    
    for var in variables:
        if var+'_int' not in dataset or force_copy:
            print(f'Creating/forcing new variable {var}_int')
            dataset[var+'_int'] = xr.DataArray(
                data=np.full_like(dataset[var].values, np.nan),
                dims=dataset[var].dims,
                coords=dataset[var].coords
            )
        # Loop over each time step
        for time_index in range(dataset.dims['time']):

            print(var, time_index)# Perform linear interpolation for the current time step
            try:
                is2_to_interp = dataset[var].isel(time=time_index).values
                is2_to_interp[np.where(dataset.sea_ice_conc.isel(time=time_index) < 0.15)] = 0  # Set 15% conc or less to 0 thickness
                np_interpolated = griddata((xptsIS2[(np.isfinite(is2_to_interp))], 
                                            yptsIS2[(np.isfinite(is2_to_interp))]), 
                                            is2_to_interp[(np.isfinite(is2_to_interp))].flatten(),
                                            (xptsIS2, yptsIS2), 
                                            fill_value=np.nan,
                                            method=method)
                np_interpolated[~(np.isfinite(dataset.sea_ice_conc.isel(time=time_index)))] = np.nan  # Remove thickness data where cdr data is nan 
                np_interpolated[np.where(dataset.sea_ice_conc.isel(time=time_index) < 0.5)] = np.nan  # Remove thickness data where cdr data < 50% concentration

                x_stddev = 0.5
                kernel = Gaussian2DKernel(x_stddev=x_stddev)
                np_interpolated_gauss = convolve(np_interpolated, kernel)
                np_interpolated_gauss[~(np.isfinite(dataset.sea_ice_conc.isel(time=time_index)))] = np.nan  # Remove thickness data where cdr data is nan 
                np_interpolated_gauss[np.where(dataset.sea_ice_conc.isel(time=time_index) < 0.5)] = np.nan  # Remove thickness data where cdr data < 50% concentration
                
                #print(np_interpolated_gauss.shape)
                # Ensure the data is in the correct shape
                #np_interpolated_gauss = np_interpolated_gauss[np.newaxis, :, :]  # Add a new axis for time

                # Add the interpolated data to the new dataset
                # Update the data using loc indexer
                #dataset[var+'_int'] = dataset[var+'_int'].copy(deep=True)
                dataset[var+'_int'][dict(time=time_index)] = np_interpolated_gauss
                print('Interpolated.')

            except:
                print('no data or issue with gridding, so skipping...')
                dataset[var+'_int'].isel(time=time_index)[:] = np.full(dataset[var+'_int'].isel(time=time_index)[:].shape, np.nan)
                continue
    print('done, returning new dataset')
    return dataset

def apply_interpolation_timestep(IS2_CS2_allseason, xptsIS2, yptsIS2, variables):
    # Apply interpolation to all data for consistency
    # Interpolation settings 
    method = "linear" 

    # Create a new dataset to store the interpolated values
    IS2_CS2_allseason_int = IS2_CS2_allseason.copy(deep=True)

    for var in variables:
        print(var)# Perform linear interpolation for the current time step
        #try:
        is2_to_interp = IS2_CS2_allseason[var].values.copy()
        is2_to_interp[np.where(IS2_CS2_allseason.sea_ice_conc < 0.15)] = 0  # Set 15% conc or less to 0 thickness
        np_interpolated = griddata((xptsIS2[(np.isfinite(is2_to_interp))], 
                                    yptsIS2[(np.isfinite(is2_to_interp))]), 
                                    is2_to_interp[(np.isfinite(is2_to_interp))].flatten(),
                                    (xptsIS2, yptsIS2), 
                                    fill_value=np.nan,
                                    method=method)
        np_interpolated[~(np.isfinite(IS2_CS2_allseason.sea_ice_conc))] = np.nan  # Remove thickness data where cdr data is nan 
        np_interpolated[np.where(IS2_CS2_allseason.sea_ice_conc < 0.5)] = np.nan  # Remove thickness data where cdr data < 50% concentration

        x_stddev = 0.5
        kernel = Gaussian2DKernel(x_stddev=x_stddev)
        np_interpolated_gauss = convolve(np_interpolated, kernel)
        np_interpolated_gauss[~(np.isfinite(IS2_CS2_allseason.sea_ice_conc))] = np.nan  # Remove thickness data where cdr data is nan 
        np_interpolated_gauss[np.where(IS2_CS2_allseason.sea_ice_conc < 0.5)] = np.nan  # Remove thickness data where cdr data < 50% concentration
        print('Interpolated.')
        print(np_interpolated_gauss.shape)
        # Ensure the data is in the correct shape
        #np_interpolated_gauss = np_interpolated_gauss[np.newaxis, :, :]  # Add a new axis for time

        # Add the interpolated data to the new dataset
        IS2_CS2_allseason_int[var+'_int'][:] = np_interpolated_gauss
        #except:
        #    print('no data or issue with gridding, so skipping...')
        #    #IS2SITMOGR4_v3[var].isel(time=time_index)[:] = np.full(np_interpolated_gauss.shape, np.nan)
        #    continue
    return IS2_CS2_allseason_int