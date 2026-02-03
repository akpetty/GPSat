#!/usr/bin/env python3
"""
Combine monthly IS2_interp_test_petty_YYYY-MM-DD.nc files into a single cleaned NetCDF.

- Looks in data_dir for subdirs named {run_string}_{YYYYMMDD}_{version_string} (e.g. run_30days_smap_20181215_v01), finds the NC file(s) in each, and concatenates along the time dimension. Saves combined NetCDF and browse images in data_dir by default.
- Adds grid-cell area from NSIDC0771 (e.g. NSIDC0771_CellArea_PS_N25km_v1.1.nc).
- Adds middle-day (15th) SIC from CDR: path is .../monthly/final/v6/ ; script adds year and finds daily file in that folder.
- All inputs are assumed to be on the same 2D grid (no regridding or padding).
- NetCDF metadata follows gen_IS2SITMOGR4_V4.py conventions.
- Requires NSIDC-0780 region mask (sea_ice_region_surface_mask); masks out CAA (value 12) in ice thickness, volume, and uncertainty only.
- CDR concentration and region mask are flipped (flipud) when loading so they match grid orientation.
- Optional: --browse_dir writes one V4-style browse image per month (thickness, concentration, volume, uncertainty).

Usage:
    module load gcc/12.1.0 
    conda activate is2_39_gp
  python combine_monthly_netcdf.py --data_dir ../output/dev_v1/thickness/

Provide --data_dir (e.g. .../output/dev_v1/thickness/); script looks for subdirs run_30days_smap_YYYYMMDD_v01 and saves combined NC and browse images in data_dir.
Cell area and CDR SIC paths are hardcoded (ADAPT: panfs/ccds02/home/aapetty/nobackup_symlink/Data/...).
"""

import argparse
import os
import re
import glob
import numpy as np
import xarray as xr
import numpy.ma as ma
from datetime import datetime
from netCDF4 import Dataset as nc4, date2num

# Hardcoded paths (ADAPT)
CELL_AREA_PATH = "/panfs/ccds02/home/aapetty/nobackup_symlink/Data/Other/NSIDC0771_CellArea_PS_N25km_v1.1.nc"
CDR_CONC_PATH = "/panfs/ccds02/home/aapetty/nobackup_symlink/Data/ICECONC/CDR/monthly/final/v6/"
REGION_MASK_PATH = os.path.join(os.path.dirname(CELL_AREA_PATH), "NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc")
# NSIDC-0780 sea_ice_region_surface_mask: 12 = Canadian Archipelago (CAA); we mask out CAA in output
CAA_REGION_INDEX = 12

CRS_ATTRS = {
    "long_name": "NSIDC Sea Ice Polar Stereographic North",
    "grid_mapping_name": "polar_stereographic",
    "srid": "urn:ogc:def:crs:EPSG::3411",
    "units": "meters",
    "proj4text": "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs",
    "GeoTransform": "-3850000.0 25000.0 0 5850000.0 0 -25000.0",
    "latitude_of_projection_origin": 90.0,
    "standard_parallel": 70.0,
    "straight_vertical_longitude_from_pole": -45.0,
    "false_easting": 0.0,
    "false_northing": 0.0,
}


def parse_date_from_filename(path):
    """Extract date from IS2_interp_test_petty_YYYY-MM-DD.nc or YYYYMMDD."""
    basename = os.path.basename(path)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", basename)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4})(\d{2})(\d{2})", basename)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def collect_monthly_files(data_dir, run_string, version_string, file_pattern="IS2_interp_test_petty_*.nc"):
    """
    Find monthly NetCDF files by looking for subdirs named {run_string}_{YYYYMMDD}_{version_string}.
    In each subdir, looks for file matching file_pattern (e.g. IS2_interp_test_petty_2018-11-15.nc).
    Returns (list of file paths, list of dates) sorted by date; date is first day of that month.
    """
    prefix = run_string + "_"
    suffix = "_" + version_string
    dated = []
    data_dir = os.path.abspath(data_dir)
    try:
        entries = os.listdir(data_dir)
    except OSError as e:
        return [], []
    for name in sorted(entries):
        subdir = os.path.join(data_dir, name)
        if not os.path.isdir(subdir):
            continue
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        middle = name[len(prefix) : -len(suffix)]
        if len(middle) != 8 or not middle.isdigit():
            continue
        year, month, day = int(middle[:4]), int(middle[4:6]), int(middle[6:8])
        d = datetime(year, month, 1)
        full_pattern = os.path.join(subdir, file_pattern)
        candidates = glob.glob(full_pattern)
        if not candidates:
            candidates = glob.glob(os.path.join(subdir, "*.nc"))
        if not candidates:
            continue
        f = candidates[0]
        if len(candidates) > 1:
            mon_str = "{:04d}{:02d}".format(year, month)
            for c in candidates:
                if mon_str in os.path.basename(c):
                    f = c
                    break
        dated.append((d, f))
    dated.sort(key=lambda x: x[0])
    return [p for _, p in dated], [d for d, _ in dated]


def load_region_mask(path, ny, nx):
    """
    Load NSIDC-0780 sea ice region mask from NetCDF (sea_ice_region_surface_mask).
    Returns (region_mask_2d, caa_mask_2d) where caa_mask is True for Canadian Archipelago (value 12).
    """
    if path.endswith(".nc"):
        ds = xr.open_dataset(path)
        if "sea_ice_region_surface_mask" in ds:
            r = ds["sea_ice_region_surface_mask"].values
        elif "region_mask" in ds:
            r = ds["region_mask"].values
        else:
            r = ds[list(ds.data_vars)[0]].values
        ds.close()
    else:
        # .bin format: uint8, [448, 304]
        with open(path, "rb") as fd:
            r = np.fromfile(fd, dtype=np.uint8)
        r = np.reshape(r, [448, 304])
    # Same orientation as concentration: flip so it matches NSIDC/IS2 grid
    r = np.flipud(np.asarray(r, dtype=np.float64))
    r = ensure_shape_2d(r, ny, nx)
    caa_mask = (r == CAA_REGION_INDEX)
    # Keep integer type; use -9999 for missing/invalid
    r_int = np.where(np.isfinite(r), np.asarray(r, dtype=np.int16), np.int16(-9999))
    return r_int, caa_mask


def load_cell_area(nc_path):
    """Load grid cell area from NSIDC NetCDF file."""
    ds = xr.open_dataset(nc_path)
    if "cell_area" in ds:
        area = ds["cell_area"]
    else:
        keys = [k for k in ds.data_vars]
        if not keys:
            raise ValueError("No data variable in cell area file: " + nc_path)
        area = ds[keys[0]]
    return area.squeeze(), ds


def load_cdr_sic_middle_day(cdr_conc_path, year, month):
    """
    Load SIC for the middle day (15th) of the month from daily files.
    Path: {cdr_conc_path}/{year}/, file matching *YYYYMM15*.nc or similar.
    CDR path should be .../CDR/monthly/final/v6/ ; script appends the year folder.
    """
    year_str = str(year)
    mon_str = "{:02d}".format(month)
    # Middle day of month (15th)
    mid_day = 15
    day_str = "{:02d}".format(mid_day)
    # CDR path is .../v6/ ; add year
    base = os.path.join(cdr_conc_path, year_str)
    if not os.path.isdir(base):
        return None
    # Match daily file: *YYYYMMDD*.nc
    date_str = year_str + mon_str + day_str
    pattern = os.path.join(base, "*" + date_str + "*.nc")
    files = glob.glob(pattern)
    if not files:
        # Try without date in name, or YYYY-MM-DD
        alt = os.path.join(base, "*" + year_str + "-" + mon_str + "-" + day_str + "*.nc")
        files = glob.glob(alt)
    if not files:
        return None
    path = sorted(files)[0]
    try:
        with nc4(path, "r") as f:
            for vname in ("cdr_seaice_conc", "sea_ice_conc", "seaice_conc_cdr", "concentration", "sic"):
                if vname in f.variables:
                    arr = f.variables[vname]
                    if arr.ndim == 3:
                        arr = arr[0]
                    arr = np.asarray(arr, dtype=np.float32)
                    # CDR grid is top-down; flip so it matches NSIDC/IS2 grid orientation
                    return np.flipud(arr)
        return None
    except Exception:
        return None


def ensure_shape_2d(arr, ny, nx):
    """Ensure 2D array is (ny, nx); transpose if (nx, ny). Same grid assumed, no padding."""
    a = np.asarray(arr, dtype=np.float32)
    if a.shape == (ny, nx):
        return a
    if a.shape == (nx, ny):
        return np.transpose(a)
    raise ValueError("Array shape {} does not match grid (ny={}, nx={})".format(a.shape, ny, nx))


def plot_browse_month_v0(combined, time_idx, save_path, fill_value=-999.0):
    """
    Browse plot v0: one image per month, V4 IS2SITMOGR4 style.
    Panels: (a) ice thickness, (b) sea ice concentration, (c) volume per grid-cell area,
    (d) ice thickness uncertainty (if present), (e) region mask, (f) grid cell area.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colorbar as mcbar
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.feature import NaturalEarthFeature

    proj = ccrs.NorthPolarStereo(central_longitude=-45)
    land_10m = NaturalEarthFeature("physical", "land", "10m", facecolor="0.95", edgecolor="0.5")
    lon = combined.longitude.values if "longitude" in combined else combined.lon.values
    lat = combined.latitude.values if "latitude" in combined else combined.lat.values

    t = combined.time.values[time_idx]
    dt = datetime.utcfromtimestamp(t.astype("datetime64[s]").astype(float))
    mon_str = dt.strftime("%B %Y")
    date_str = dt.strftime("%Y%m")

    vars_list = []
    labels = []
    cbar_labels = []
    vmin_list = []
    vmax_list = []
    cmaps = []

    # (a) ice thickness
    th = np.ma.masked_where(
        ~np.isfinite(combined.ice_thickness.values[time_idx]) | (combined.ice_thickness.values[time_idx] == fill_value),
        combined.ice_thickness.values[time_idx],
    )
    vars_list.append(th)
    labels.append("sea ice thickness")
    cbar_labels.append("ice thickness (m)")
    vmin_list.append(0)
    vmax_list.append(5)
    cmaps.append(plt.cm.viridis)

    # (b) sea ice concentration
    conc = np.ma.masked_where(
        ~np.isfinite(combined.sea_ice_conc.values[time_idx]) | (combined.sea_ice_conc.values[time_idx] == fill_value),
        combined.sea_ice_conc.values[time_idx],
    )
    vars_list.append(conc)
    labels.append("sea ice concentration")
    cbar_labels.append("concentration")
    vmin_list.append(0)
    vmax_list.append(1)
    cmaps.append(plt.cm.Blues_r)

    # (c) volume per grid-cell area = thickness / concentration (m); masked where SIC < 0.15
    th = combined.ice_thickness.values[time_idx].astype(float)
    conc = combined.sea_ice_conc.values[time_idx].astype(float)
    conc_safe = np.where(np.isfinite(conc) & (conc > 1e-6), conc, np.nan)
    vol_per_area = np.ma.masked_where(
        ~np.isfinite(th) | (th == fill_value) | ~np.isfinite(conc_safe) | (conc < 0.15),
        th / conc_safe,
    )
    vars_list.append(vol_per_area)
    labels.append("volume per grid-cell area (where sic>15%)")
    cbar_labels.append("ice thickness (m)")
    vmin_list.append(0)
    vmax_list.append(5)
    cmaps.append(plt.cm.viridis)

    # (d) uncertainty if present
    if "ice_thickness_unc" in combined:
        unc = np.ma.masked_where(
            ~np.isfinite(combined.ice_thickness_unc.values[time_idx])
            | (combined.ice_thickness_unc.values[time_idx] == fill_value),
            combined.ice_thickness_unc.values[time_idx],
        )
        vars_list.append(unc)
        labels.append("thickness uncertainty")
        cbar_labels.append("uncertainty (m)")
        vmin_list.append(0)
        vmax_list.append(0.5)
        cmaps.append(plt.cm.YlOrRd)

    # (e) region mask (2D)
    if "region_mask" in combined:
        rm = np.ma.masked_where(
            combined.region_mask.values == -9999,
            combined.region_mask.values.astype(float),
        )
        vars_list.append(rm)
        labels.append("region mask")
        cbar_labels.append("region index")
        vmin_list.append(0)
        vmax_list.append(32)
        cmaps.append(plt.cm.nipy_spectral)

    # (f) grid cell area (2D)
    if "grid_cell_area" in combined:
        gca = combined.grid_cell_area.values.astype(float)
        gca = np.ma.masked_where(~np.isfinite(gca) | (gca == fill_value) | (gca <= 0), gca)
        vars_list.append(gca)
        labels.append("grid cell area")
        cbar_labels.append(r"area (m$^2$)")
        vmin_list.append(0)
        vmax_list.append(float(np.percentile(gca.compressed(), 99)) if gca.count() > 0 else 1e7)
        cmaps.append(plt.cm.YlGn)

    npanels = len(vars_list)
    ncols = 2
    nrows = (npanels + 1) // 2
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(7, 5 * nrows), subplot_kw={"projection": proj}
    )
    axs = np.atleast_2d(axs)
    plt.subplots_adjust(bottom=0.06, left=0.02, top=0.94, right=0.98, wspace=0.01, hspace=0.06)

    sic_bg = np.ma.masked_where(conc < 0.15, conc)
    for i in range(npanels):
        ax = axs.flat[i]
        plt.sca(ax)
        # Background: low SIC in gray (V4 style)
        ax.pcolormesh(
            lon, lat, np.ma.filled(sic_bg, np.nan), vmin=0, vmax=2, cmap=plt.cm.gray_r,
            transform=ccrs.PlateCarree(), zorder=1,
        )
        im = ax.pcolormesh(
            lon, lat, vars_list[i],
            vmin=vmin_list[i], vmax=vmax_list[i], cmap=cmaps[i],
            transform=ccrs.PlateCarree(), zorder=2,
        )
        ax.set_extent([-179, 179, 50, 90], ccrs.PlateCarree())
        #ax.coastlines("10m", linewidth=0.3, zorder=3)
        ax.add_feature(land_10m, zorder=3)
        ax.gridlines(draw_labels=False, linewidth=0.22, color="gray", alpha=0.5, linestyle="--")
        ax.annotate(
            "(" + chr(97 + i) + ") " + labels[i],
            xy=(0.02, 1.03), xycoords="axes fraction", fontsize=9, zorder=10,
        )
        if i == 0:
            ax.annotate(
                mon_str, xy=(0.98, 1.03), xycoords="axes fraction",
                horizontalalignment="right", fontsize=9, zorder=10,
            )
        cax, kw = mcbar.make_axes(ax, location="bottom", pad=0.04, shrink=0.5, fraction=0.10)
        cb = fig.colorbar(im, cax=cax, extend="both", **kw)
        cb.set_label(cbar_labels[i])
        if vmax_list[i] == 5 and vmin_list[i] == 0 and "thickness" in cbar_labels[i].lower():
            cb.set_ticks([0, 1, 2, 3, 4, 5])
        elif vmax_list[i] == 0.5:
            cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    for j in range(npanels, axs.size):
        axs.flat[j].set_visible(False)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_browse_month_v1(combined, time_idx, save_path, fill_value=-999.0):
    """
    Browse plot v1: one row of 3 panels — (a) sea ice concentration, (b) ice thickness,
    (c) thickness uncertainty.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colorbar as mcbar
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.feature import NaturalEarthFeature

    proj = ccrs.NorthPolarStereo(central_longitude=-45)
    land_10m = NaturalEarthFeature("physical", "land", "10m", facecolor="0.95", edgecolor="0.5")
    lon = combined.longitude.values if "longitude" in combined else combined.lon.values
    lat = combined.latitude.values if "latitude" in combined else combined.lat.values

    t = combined.time.values[time_idx]
    dt = datetime.utcfromtimestamp(t.astype("datetime64[s]").astype(float))
    mon_str = dt.strftime("%B %Y")
    date_str = dt.strftime("%Y%m")

    vars_list = []
    labels = []
    cbar_labels = []
    vmin_list = []
    vmax_list = []
    cmaps = []

    # (a) sea ice concentration
    conc = np.ma.masked_where(
        ~np.isfinite(combined.sea_ice_conc.values[time_idx]) | (combined.sea_ice_conc.values[time_idx] == fill_value),
        combined.sea_ice_conc.values[time_idx],
    )
    vars_list.append(conc)
    labels.append("sea ice concentration")
    cbar_labels.append("concentration")
    vmin_list.append(0)
    vmax_list.append(1)
    cmaps.append(plt.cm.Blues_r)

    # (b) ice thickness
    th = np.ma.masked_where(
        ~np.isfinite(combined.ice_thickness.values[time_idx]) | (combined.ice_thickness.values[time_idx] == fill_value),
        combined.ice_thickness.values[time_idx],
    )
    vars_list.append(th)
    labels.append("sea ice thickness")
    cbar_labels.append("ice thickness (m)")
    vmin_list.append(0)
    vmax_list.append(4)
    cmaps.append(plt.cm.viridis)

    # (c) thickness uncertainty (if present)
    if "ice_thickness_unc" in combined:
        unc = np.ma.masked_where(
            ~np.isfinite(combined.ice_thickness_unc.values[time_idx])
            | (combined.ice_thickness_unc.values[time_idx] == fill_value),
            combined.ice_thickness_unc.values[time_idx],
        )
        vars_list.append(unc)
        labels.append("thickness uncertainty")
        cbar_labels.append("uncertainty (m)")
        vmin_list.append(0)
        vmax_list.append(0.5)
        cmaps.append(plt.cm.YlOrRd)

    npanels = len(vars_list)
    if npanels == 0:
        plt.close()
        return
    fig, axs = plt.subplots(1, 3, figsize=(9, 4), subplot_kw={"projection": proj})
    plt.subplots_adjust(bottom=0.06, left=0.02, top=0.94, right=0.98, wspace=0.01)

    sic_bg = np.ma.masked_where(conc < 0.15, conc)
    for i in range(npanels):
        ax = axs.flat[i]
        plt.sca(ax)
        if i == 0:
            ax.pcolormesh(
                lon, lat, np.ma.filled(sic_bg, np.nan), vmin=0, vmax=2, cmap=plt.cm.gray_r,
                transform=ccrs.PlateCarree(), zorder=1,
            )
        im = ax.pcolormesh(
            lon, lat, vars_list[i],
            vmin=vmin_list[i], vmax=vmax_list[i], cmap=cmaps[i],
            transform=ccrs.PlateCarree(), zorder=2,
        )
        ax.set_extent([-179, 179, 50, 90], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.95", edgecolor="black", linewidth=0.05, zorder=3)
        ax.gridlines(draw_labels=False, linewidth=0.22, color="gray", alpha=0.5, linestyle="--")
        ax.annotate(
            "(" + chr(97 + i) + ") " + labels[i],
            xy=(0.02, 1.03), xycoords="axes fraction", fontsize=9, zorder=10,
        )
        if i == 0:
            ax.annotate(
                mon_str, xy=(0.98, 1.03), xycoords="axes fraction",
                horizontalalignment="right", fontsize=9, zorder=10,
            )
        cax, kw = mcbar.make_axes(ax, location="bottom", pad=0.02, shrink=0.7, fraction=0.10)
        cb = fig.colorbar(im, cax=cax, extend="both", **kw)
        cb.set_label(cbar_labels[i])
        if vmax_list[i] == 5 and vmin_list[i] == 0 and "thickness" in cbar_labels[i].lower():
            cb.set_ticks([0, 1, 2, 3, 4])
        elif vmax_list[i] == 0.5:
            cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    for j in range(npanels, 3):
        axs.flat[j].set_visible(False)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def plot_seasonal_cycle(combined, save_path, fill_value=-999.0):
    """
    Generate a browse image of the seasonal cycle of main quantities, masked to
    regions 1–7 only: area-weighted mean thickness, mean concentration,
    total sea ice volume (km³), and mean ice volume per grid-cell area (m).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    times = combined.time.values
    dates = [datetime.utcfromtimestamp(t.astype("datetime64[s]").astype(float)) for t in times]

    # Mask: keep only regions 1–7 (Central Arctic through Barents Sea)
    region_mask = combined.region_mask.values
    valid_1_7 = (region_mask >= 1) & (region_mask <= 7)
    valid_1_7_3d = np.broadcast_to(valid_1_7, (len(times),) + valid_1_7.shape)

    # Area-weighted mean sea ice thickness (m) over regions 1–7 only
    th = combined.ice_thickness.values
    area = combined.grid_cell_area.values
    th_valid = np.isfinite(th) & (th != fill_value) & valid_1_7_3d
    weighted_sum = np.where(th_valid, th * area, 0.0).sum(axis=(1, 2))
    total_area = np.where(th_valid, area, 0.0).sum(axis=(1, 2))
    mean_th = np.where(total_area > 0, weighted_sum / total_area, np.nan)

    # Mean sea ice concentration over regions 1–7 only
    conc = combined.sea_ice_conc.values
    conc_valid = np.where(np.isfinite(conc) & (conc != fill_value) & valid_1_7_3d, conc, np.nan)
    mean_conc = np.nanmean(conc_valid, axis=(1, 2))

    # Total sea ice volume (km³) — all regions (from dataset)
    total_vol = combined.total_sea_ice_volume.values

    # Mean ice volume per grid-cell area (m) over regions 1–7 only
    vpa = combined.ice_volume_per_grid_cell_area.values
    vpa_valid = np.where(np.isfinite(vpa) & (vpa != fill_value) & valid_1_7_3d, vpa, np.nan)
    mean_vpa = np.nanmean(vpa_valid, axis=(1, 2))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Seasonal cycle (regions 1–7 only) — IS2-SMOS-SMAP combined thickness dataset, v0", fontsize=12)

    # Same y-scale (0–3 m) for thickness and volume-per-area for direct comparison
    ylim_m = (0, 3)

    axs[0, 0].plot(dates, mean_th, "C0-o", markersize=4)
    axs[0, 0].set_ylabel("m")
    axs[0, 0].set_title("Mean sea ice thickness (area-weighted)")
    axs[0, 0].set_ylim(ylim_m)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axs[0, 1].plot(dates, mean_conc, "C1-o", markersize=4)
    axs[0, 1].set_ylabel("1")
    axs[0, 1].set_title("Mean sea ice concentration")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axs[1, 0].plot(dates, total_vol, "C2-o", markersize=4)
    axs[1, 0].set_ylabel("km³")
    axs[1, 0].set_title("Total sea ice volume (all regions)")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    axs[1, 1].plot(dates, mean_vpa, "C3-o", markersize=4)
    axs[1, 1].set_ylabel("m")
    axs[1, 1].set_title("Mean ice volume per grid-cell area (sic>15%)")
    axs[1, 1].set_ylim(ylim_m)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    for ax in axs.flat:
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Combine monthly IS2 NetCDFs and add grid area + CDR SIC."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing run subdirs; combined NC and browse images are saved here by default.")
    parser.add_argument("--output", type=str, default=None, help="Output path for combined NetCDF (default: {data_dir}/is2smgp_sit_v0.nc).")
    parser.add_argument("--run_string", type=str, default="run_30days_smap", help="Run name prefix; subdirs must be {run_string}_YYYYMMDD_{version_string}.")
    parser.add_argument("--version_string", type=str, default="v01", help="Version suffix; subdirs must be {run_string}_YYYYMMDD_{version_string}.")
    parser.add_argument("--pattern", type=str, default="IS2_interp_test_petty_*.nc", help="Glob pattern for NC file inside each run subdir (default: IS2_interp_test_petty_*.nc).")
    parser.add_argument("--fill_value", type=float, default=-999.0, help="Fill value for missing data.")
    parser.add_argument("--browse_dir", type=str, default=None, help="Directory for browse images (default: data_dir). Set to empty to skip browse.")
    parser.add_argument("--browse_v0", action="store_true", help="Also produce v0 browse images (6 panels). Default: only v1 and seasonal cycle.")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    if args.output is None:
        args.output = os.path.join(args.data_dir, "is2smgp_sit_v0.nc")
    if args.browse_dir is None:
        args.browse_dir = args.data_dir

    files, dates = collect_monthly_files(
        args.data_dir, args.run_string, args.version_string, args.pattern
    )
    if not files:
        raise SystemExit(
            "No run subdirs found in "
            + args.data_dir
            + " matching "
            + args.run_string
            + "_YYYYMMDD_"
            + args.version_string
            + " with files matching "
            + args.pattern
            + " inside. Check that --data_dir is correct (use absolute path if needed)."
        )

    print("Found {} monthly files.".format(len(files)))
    ds_list = []
    for f, d in zip(files, dates):
        ds = xr.open_dataset(f)
        if "time" in ds.sizes and ds.sizes["time"] > 1:
            ds = ds.isel(time=0, drop=True)
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")
        ds["time"] = [np.datetime64(d, "ns")]
        ds_list.append(ds)

    combined = xr.concat(ds_list, dim="time")
    combined = combined.sortby("time")

    if "SIT" in combined and "ice_thickness" not in combined:
        combined = combined.rename({"SIT": "ice_thickness"})
    if "SIT_var" in combined and "ice_thickness_unc" not in combined:
        combined = combined.rename({"SIT_var": "ice_thickness_unc"})
    # Input uncertainty is variance (f*_var); convert to standard deviation (m)
    if "ice_thickness_unc" in combined:
        combined["ice_thickness_unc"].values[:] = np.sqrt(np.maximum(combined.ice_thickness_unc.values.astype(float), 0.0))

    ny, nx = combined.lat.shape

    # Load grid cell area (same 2D grid as monthly files)
    print("Loading grid cell area...")
    area_da, _ = load_cell_area(CELL_AREA_PATH)
    grid_cell_area = ensure_shape_2d(area_da.values, ny, nx)
    combined["grid_cell_area"] = (("y", "x"), grid_cell_area)

    # Load CDR SIC per month (same 2D grid)
    sic_list = []
    for t in combined.time.values:
        dt = datetime.utcfromtimestamp(t.astype("datetime64[s]").astype(float))
        if os.path.isdir(CDR_CONC_PATH):
            arr = load_cdr_sic_middle_day(CDR_CONC_PATH, dt.year, dt.month)
            if arr is not None:
                arr = ensure_shape_2d(arr, ny, nx)
                arr = np.where(np.isfinite(arr) & (arr >= 0) & (arr <= 1.1), arr, np.nan)
            else:
                arr = np.full((ny, nx), np.nan, dtype=np.float32)
        else:
            arr = np.full((ny, nx), np.nan, dtype=np.float32)
        sic_list.append(arr)
    combined["sea_ice_conc"] = (("time", "y", "x"), np.array(sic_list, dtype=np.float32))

    # Sea ice volume per grid cell = thickness × grid_cell_area, in km³ (divide m³ by 1e9)
    vol_m3_per_cell = combined.ice_thickness.values * grid_cell_area[np.newaxis, :, :]
    combined["sea_ice_volume"] = (("time", "y", "x"), np.asarray(vol_m3_per_cell / 1.0e9, dtype=np.float32))

    # Load region mask (IS2SITMOGR4 style) and mask out CAA (Canadian Archipelago) — required
    print("Loading region mask and masking CAA...")
    region_mask_2d, caa_mask = load_region_mask(REGION_MASK_PATH, ny, nx)
    combined["region_mask"] = (("y", "x"), region_mask_2d)
    # Mask out CAA (region 12) in thickness, volume, and uncertainty only (not concentration)
    caa_3d = np.broadcast_to(caa_mask, (len(combined.time), ny, nx))
    combined["ice_thickness"].values[:] = np.where(caa_3d, np.nan, combined.ice_thickness.values)
    combined["sea_ice_volume"].values[:] = np.where(caa_3d, np.nan, combined.sea_ice_volume.values)
    if "ice_thickness_unc" in combined:
        combined["ice_thickness_unc"].values[:] = np.where(caa_3d, np.nan, combined.ice_thickness_unc.values)

    # Ice volume per grid-cell area = thickness / concentration (m); masked where SIC < 0.15
    conc_3d = combined.sea_ice_conc.values
    th_3d = combined.ice_thickness.values
    conc_safe = np.where(np.isfinite(conc_3d) & (conc_3d > 1e-6), conc_3d, np.nan)
    vol_per_area = np.where(np.isfinite(conc_safe), th_3d / conc_safe, np.nan)
    vol_per_area = np.where((conc_3d >= 0.15) & np.isfinite(vol_per_area), vol_per_area, np.nan)
    combined["ice_volume_per_grid_cell_area"] = (("time", "y", "x"), np.asarray(vol_per_area, dtype=np.float32))

    # Total sea ice volume: sum of sea_ice_volume over all grid cells (already in km³ per cell)
    total_vol_km3 = np.nansum(np.where(np.isfinite(combined.sea_ice_volume.values), combined.sea_ice_volume.values, 0.0), axis=(1, 2))
    combined["total_sea_ice_volume"] = (("time",), total_vol_km3.astype(np.float32))

    # Area-weighted mean sea ice thickness (xarray): sum(th * area) / sum(area) over valid cells
    # Mask out regions 1–7 (Central Arctic through Barents Sea) so mean is over remaining regions only
    valid_region = (combined.region_mask < 1) | (combined.region_mask > 7)
    valid = (
        combined.ice_thickness.notnull()
        & np.isfinite(combined.ice_thickness)
        & (combined.ice_thickness != args.fill_value)
        & valid_region
    )
    weighted_sum = (combined.ice_thickness * combined.grid_cell_area).where(valid).sum(dim=["y", "x"])
    total_area = combined.grid_cell_area.where(valid).sum(dim=["y", "x"])
    mean_thickness_area_weighted = weighted_sum / total_area
    combined["mean_sea_ice_thickness_area_weighted"] = mean_thickness_area_weighted.astype(np.float32)

    # Browse images: v1 (3 panels) and seasonal cycle by default; v0 (6 panels) only with --browse_v0
    out_path = args.output
    odir = os.path.dirname(os.path.abspath(out_path))
    if odir:
        os.makedirs(odir, exist_ok=True)
    fv = args.fill_value

    if args.browse_dir:
        os.makedirs(args.browse_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(out_path))[0]
        seasonal_path = os.path.join(args.browse_dir, base_name + "_browse_seasonal_cycle.png")
        plot_seasonal_cycle(combined, seasonal_path, fv)
        print("Browse: " + seasonal_path)
        for ti in range(len(combined.time)):
            t = combined.time.values[ti]
            dt = datetime.utcfromtimestamp(t.astype("datetime64[s]").astype(float))
            date_str = dt.strftime("%Y%m")
            if args.browse_v0:
                save_path_v0 = os.path.join(args.browse_dir, base_name + "_browse_v0_" + date_str + ".png")
                plot_browse_month_v0(combined, ti, save_path_v0, fv)
                print("Browse v0: " + save_path_v0)
            save_path_v1 = os.path.join(args.browse_dir, base_name + "_browse_v1_" + date_str + ".png")
            plot_browse_month_v1(combined, ti, save_path_v1, fv)
            print("Browse v1: " + save_path_v1)

    # Write NetCDF with CF-style metadata (gen_IS2SITMOGR4_V4 style)
    nc = nc4(out_path, "w", format="NETCDF4")
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)
    nc.createDimension("time", len(combined.time))

    proj = nc.createVariable("crs", "i4")
    for k, v in CRS_ATTRS.items():
        setattr(proj, k, v)

    time_var = nc.createVariable("time", "f8", ("time",))
    time_var.units = "days since 1970-01-01"
    time_var.calendar = "gregorian"
    time_var.long_name = "time"
    time_var.standard_name = "time"
    times_dt = [datetime.utcfromtimestamp(t.astype("datetime64[s]").astype(float)) for t in combined.time.values]
    time_var[:] = date2num(times_dt, units=time_var.units, calendar=time_var.calendar)

    lon = nc.createVariable("longitude", "f4", ("y", "x"))
    lat = nc.createVariable("latitude", "f4", ("y", "x"))
    xgrid = nc.createVariable("x", "f4", ("x",))
    ygrid = nc.createVariable("y", "f4", ("y",))
    lon.units = "degrees_east"
    lon.long_name = "longitude"
    lat.units = "degrees_north"
    lat.long_name = "latitude"
    xgrid.units = "meters"
    xgrid.long_name = "projection x coordinate"
    ygrid.units = "meters"
    ygrid.long_name = "projection y coordinate"

    lon[:] = combined.longitude.values if "longitude" in combined else combined.lon.values
    lat[:] = combined.latitude.values if "latitude" in combined else combined.lat.values
    xvals = combined.x.values
    yvals = combined.y.values
    if getattr(xvals, "ndim", 1) == 1 and len(xvals) == nx and len(yvals) == ny:
        xgrid[:] = xvals
        ygrid[:] = yvals
    else:
        xgrid[:] = xvals[0, :] if xvals.ndim == 2 else xvals.flat[:nx]
        ygrid[:] = yvals[:, 0] if yvals.ndim == 2 else yvals.flat[:ny]

    ice_thickness = nc.createVariable("ice_thickness", "f4", ("time", "y", "x"), fill_value=fv)
    ice_thickness.units = "meters"
    ice_thickness.long_name = "sea ice thickness"
    ice_thickness.description = "Mean sea ice thickness across the grid cell (m). From ICESat-2 GP interpolation (petty)."
    ice_thickness.grid_mapping = "crs"
    arr = combined.ice_thickness.values
    ice_thickness[:] = np.where(np.isfinite(arr), arr, fv)

    if "ice_thickness_unc" in combined:
        unc = nc.createVariable("ice_thickness_unc", "f4", ("time", "y", "x"), fill_value=fv)
        unc.units = "meters"
        unc.long_name = "sea ice thickness uncertainty (standard deviation)"
        unc.description = "Standard deviation of thickness (sqrt of predictive variance from GP)."
        unc.grid_mapping = "crs"
        unc[:] = np.where(np.isfinite(combined.ice_thickness_unc.values), combined.ice_thickness_unc.values, fv)

    if "SIT_input" in combined:
        sit_input = nc.createVariable("sit_input", "f4", ("time", "y", "x"), fill_value=fv)
        sit_input.long_name = "mean input thickness (binned along-track/SMAP)"
        sit_input.grid_mapping = "crs"
        sit_input[:] = np.where(np.isfinite(combined.SIT_input.values), combined.SIT_input.values, fv)

    ice_conc = nc.createVariable("sea_ice_conc", "f4", ("time", "y", "x"), fill_value=fv)
    ice_conc.units = "1"
    ice_conc.long_name = "sea ice concentration"
    ice_conc.description = "CDR sea ice concentration for middle day (15th) of month from daily files in CDR monthly/final/v6/YYYY/."
    ice_conc.source = "https://nsidc.org/data/G02202/"
    ice_conc.grid_mapping = "crs"
    ice_conc[:] = np.where(np.isfinite(combined.sea_ice_conc.values), combined.sea_ice_conc.values, fv)

    vol_var = nc.createVariable("sea_ice_volume", "f4", ("time", "y", "x"), fill_value=fv)
    vol_var.units = "km3"
    vol_var.long_name = "sea ice volume per grid cell"
    vol_var.description = "Total amount of sea ice in the grid cell: thickness × grid_cell_area, in km³."
    vol_var.grid_mapping = "crs"
    vol_arr = combined.sea_ice_volume.values
    vol_var[:] = np.where(np.isfinite(vol_arr), vol_arr, fv)

    vpa = nc.createVariable("ice_volume_per_grid_cell_area", "f4", ("time", "y", "x"), fill_value=fv)
    vpa.units = "m"
    vpa.long_name = "sea ice volume per grid-cell area"
    vpa.description = "Sea ice thickness / sea_ice_conc, in meters. Masked where sea ice concentration < 0.15."
    vpa.grid_mapping = "crs"
    vpa_arr = combined.ice_volume_per_grid_cell_area.values
    vpa[:] = np.where(np.isfinite(vpa_arr), vpa_arr, fv)

    total_vol = nc.createVariable("total_sea_ice_volume", "f4", ("time",))
    total_vol.units = "km3"
    total_vol.long_name = "total sea ice volume"
    total_vol.description = "Sum of sea_ice_volume over all grid cells (km³). Each grid cell: thickness × grid_cell_area, in km³."
    total_vol[:] = combined.total_sea_ice_volume.values

    mth = nc.createVariable("mean_sea_ice_thickness_area_weighted", "f4", ("time",))
    mth.units = "m"
    mth.long_name = "mean sea ice thickness (area-weighted)"
    mth.description = "Area-weighted mean sea ice thickness: sum(ice_thickness × grid_cell_area) / sum(grid_cell_area) over valid cells, in m. Masked within regions 1–7 (Central Arctic through Barents Sea)."
    mth[:] = np.where(np.isfinite(combined.mean_sea_ice_thickness_area_weighted.values), combined.mean_sea_ice_thickness_area_weighted.values, fv)

    gca = nc.createVariable("grid_cell_area", "f4", ("y", "x"), fill_value=fv)
    gca.units = "m2"
    gca.long_name = "grid cell area"
    gca.description = "Grid cell area from NSIDC Polar Stereographic North (EPSG:3411)."
    gca.source = "https://doi.org/10.5067/N6INPBT8Y104"
    gca.grid_mapping = "crs"
    gca[:] = np.where(np.isfinite(grid_cell_area), grid_cell_area, fv)

    rvar = nc.createVariable("region_mask", "i2", ("y", "x"), fill_value=np.int16(-9999))
    rvar.long_name = "Northern Hemisphere region mask"
    rvar.description = "NSIDC-0780 sea ice region mask (sea_ice_region_surface_mask). CAA (Canadian Archipelago, value 12) is masked out in ice thickness, volume, and uncertainty only."
    rvar.source = "https://doi.org/10.5067/CYW3O8ZUNIWC"
    rvar.flag_values = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 30, 31, 32"
    rvar.flag_meanings = "Outside_of_defined_regions Central_Arctic Beaufort_Sea Chukchi_Sea East_Siberian_Sea Laptev_Sea Kara_Sea Barents_Sea East_Greenland_Sea Baffin_Bay_and_Davis_Strait Gulf_of_St._Lawrence Hudson_Bay Canadian_Archipelago Bering_Sea Sea_of_Okhotsk Sea_of_Japan Bohai_Sea Gulf_of_Bothnia Baltic_Sea Gulf_of_Alaska Land Coast Lakes"
    rvar.grid_mapping = "crs"
    rvar[:] = region_mask_2d

    nc.contact = "Alek Petty (akpetty@umd.edu)"
    nc.description = (
        "Combined monthly gridded Arctic sea ice thickness from ICESat-2 GP interpolation, "
        "with grid cell area and CDR sea ice concentration. NSIDC Polar Stereographic North (EPSG:3411)."
    )
    nc.history = "Created " + datetime.today().strftime("%Y-%m-%d")
    nc.source = "Combined from monthly IS2_interp_test_petty files; grid area NSIDC0771; SIC CDR."
    nc.close()
    print("Written: " + out_path)


if __name__ == "__main__":
    main()
