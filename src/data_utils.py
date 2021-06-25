# Utilities for reading, writing, and formatting data
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import regionmask

"""
TODO:
- add check for extremely large observation error and remove those data values in preprocessing
"""

## Reading
def prep_sif(ds):
    """Preprocess an OCO-2 SIF Lite file.
    
    NOTE: 
    SIF_Uncertainty_740nm is defined as "estimated 1-sigma uncertainty of Solar Induced Fluorescence at 740 nm. Uncertainty computed from continuum level radiance at 740 nm." Squaring this value yeilds the variance of the measurement error which will be added to the diagonals of the covariance matrix.
    """

    # drop unused variables
    variable_list = [
        "Daily_SIF_740nm",
        "SIF_Uncertainty_740nm",
        "Quality_Flag",
        "Longitude",
        "Latitude",
        "Delta_Time",
    ]
    ds = ds[variable_list]

    # apply quality filters
    ds["SIF_plus_3sig"] = ds.Daily_SIF_740nm + 3 * ds.SIF_Uncertainty_740nm
    ds = ds.where(ds.Quality_Flag != 2, drop=True)
    ds = ds.where(ds.SIF_plus_3sig > 0, drop=True)

    # format dataset
    return xr.Dataset(
        {
            "sif": (["time"], ds.Daily_SIF_740nm),
            "sif_var": (["time"], ds.SIF_Uncertainty_740nm ** 2),
        },
        coords={
            "lon": (["time"], ds.Longitude),
            "lat": (["time"], ds.Latitude),
            "time": ds.Delta_Time.values,
        },
    )


def prep_xco2(ds):
    """Preprocess an OCO-2 FP Lite file.
    
    NOTE: 
    xco2_uncertainty is defined as "the posterior uncertainty in XCO2 calculated by the L2 algorithm, in ppm. This is generally 30-50% smaller than the true retrieval uncertainty." Doubling this value yields a conservative estimate of the variance of the measurement error which will be added to the diagonals of the covariance matrix.
    """

    # drop unused variables
    variable_list = [
        "xco2",
        "xco2_uncertainty",
        "xco2_quality_flag",
        "longitude",
        "latitude",
        "time",
    ]
    ds = ds[variable_list]

    # apply quality filters
    ds = ds.where(ds.xco2_quality_flag == 0, drop=True)

    # format dataset
    return xr.Dataset(
        {"xco2": (["time"], ds.xco2), "xco2_var": (["time"], ds.xco2_uncertainty * 2)},
        coords={
            "lon": (["time"], ds.longitude),
            "lat": (["time"], ds.latitude),
            "time": ds.time.values,
        },
    )


def read_transcom(path):
    """
    Read 1-degree TransCom 3 region map into xarray dataset.
    """
    ds = xr.open_dataset(path)
    ds = ds.where(ds.region < 12, drop=True)
    ds = ds.where(ds.region != 0, drop=True)
    return ds


## Formatting
def set_grid_def(lon_res=1, lat_res=1, lon_offset=0, lat_offset=0):
    assert (
        lon_offset == 0 or lat_offset == 0
    ), "lon_offset and/or lat_offset must be zero"
    return {
        "lon_res": lon_res,
        "lat_res": lat_res,
        "lon_offset": lon_offset,
        "lat_offset": lat_offset,
    }


def prep_extents(extents, grid_def):
    lon_lwr = extents[0] - grid_def["lon_res"] / 2 + grid_def["lon_offset"]
    lon_upr = extents[1] + grid_def["lon_res"] / 2 + grid_def["lon_offset"]
    lat_lwr = extents[2] - grid_def["lat_res"] / 2 + grid_def["lat_offset"]
    lat_upr = extents[3] + grid_def["lat_res"] / 2 + grid_def["lat_offset"]
    return lon_lwr, lon_upr, lat_lwr, lat_upr


def global_grid(extents=None, grid_def=None):
    """Establish longitude and latitude bins and centerpoints on a global grid."""
    if extents is None:
        extents = [-180, 180, -90, 90]
    if grid_def is None:
        grid_def = dict(lon_res=1.0, lat_res=1.0, lon_offset=0.0, lat_offset=0.0)

    lon_lwr, lon_upr, lat_lwr, lat_upr = prep_extents(extents, grid_def)
    lon_bins = np.arange(lon_lwr, lon_upr + grid_def["lon_res"], grid_def["lon_res"])
    lat_bins = np.arange(lat_lwr, lat_upr + grid_def["lat_res"], grid_def["lat_res"])
    lon_centers = (lon_bins[1:] + lon_bins[:-1]) / 2
    lat_centers = (lat_bins[1:] + lat_bins[:-1]) / 2
    return {
        "lon_bins": lon_bins,
        "lon_centers": lon_centers,
        "lat_bins": lat_bins,
        "lat_centers": lat_centers,
    }


def regrid(ds=None, df=None, extents=None, grid_def=None):
    """
    Convert dataset to dataframe and assign coordinates using a regular grid.
    """
    if ds is not None:
        df = ds.to_dataframe().reset_index()
    elif df is None:
        warnings.warn("No data provided.")

    grid = global_grid(extents=extents, grid_def=grid_def)
    bounds_check = (
        grid["lon_bins"].min() <= df.lon.min()
        and grid["lon_bins"].max() >= df.lon.max()
        and grid["lat_bins"].min() <= df.lat.min()
        and grid["lat_bins"].max() >= df.lat.max()
    )
    if not bounds_check:
        warnings.warn(
            "WARNING: dataset coordinates not within extents; may produce unexpected behavior."
        )
    # overwrite lon-lat values with grid values
    df["lon"] = pd.cut(df.lon, grid["lon_bins"], labels=grid["lon_centers"]).astype(
        float
    )
    df["lat"] = pd.cut(df.lat, grid["lat_bins"], labels=grid["lat_centers"]).astype(
        float
    )
    return df


def land_grid(extents=None, grid_def=None):
    """Collect land locations on a regular grid as an array.

    Returns rows with entries [[lat, lon]].
    """
    # establish a fine resolution grid of 0.25 degrees for accuracy
    fine_res_def = set_grid_def(lon_res=0.25, lat_res=0.25)
    grid = global_grid(extents, fine_res_def)
    land = regionmask.defined_regions.natural_earth.land_110
    mask = land.mask(grid["lon_centers"], grid["lat_centers"])
    # regrid to desired resolution and remove non-land areas
    df_mask = (
        regrid(mask, extents=extents, grid_def=grid_def)
        .dropna(subset=["region"])
        .groupby(["lon", "lat"])
        .mean()
        .reset_index()
    )
    return df_mask[["lat", "lon"]].assign(land=lambda x: 1).set_index(["lon", "lat"])


def monthly_avg(df_grid):
    """Group dataframe by relabeled lat-lon coordinates and compute monthy average."""
    return (
        df_grid.groupby(["lon", "lat"])
        .resample("1MS", on="time")
        .mean()
        .drop(columns=["lon", "lat"])
        .reset_index()
    )


def apply_land_mask(df, extents=None, grid_def=None):
    df_land = land_grid(extents, grid_def)
    return (
        df.join(df_land, on=["lon", "lat"], how="outer")
        .dropna(subset=["land"])
        .reset_index()
        .drop(columns=["land", "index"])
    )


def prep_gridded_df(ds, extents=None, grid_def=None):
    """Aggregate irregular data into a 4x5-degree grid of monthly averages over North America (land only). Return as data frame."""
    lon_lwr, lon_upr, lat_lwr, lat_upr = prep_extents(extents, grid_def)
    df = ds.to_dataframe()
    bounds = (
        (df.lon >= lon_lwr)
        & (df.lon <= lon_upr)
        & (df.lat >= lat_lwr)
        & (df.lat <= lat_upr)
    )
    # drop data outside domain extents so it's not included in edge bin averages
    df = df.loc[bounds].reset_index()
    df_grid = regrid(df=df, extents=extents, grid_def=grid_def)
    df_grid = monthly_avg(df_grid)
    return apply_land_mask(df_grid, extents, grid_def)


def augment_dataset(ds):
    """Prepare gridded dataframes for each longitude and latitude offset, and return as a single dataframe."""
    extents = [-125, -65, 22, 58]
    lat_offsets = np.linspace(-1.5, 2, 8)
    lon_offsets = np.linspace(-2, 2.5, 10)
    # drop zero offset from one set so there is no repeat of the base coordinates
    lon_offsets = lon_offsets[lon_offsets != 0]

    list_def_lat = [
        set_grid_def(lon_res=5, lat_res=4, lat_offset=lat_off)
        for lat_off in lat_offsets
    ]
    list_def_lon = [
        set_grid_def(lon_res=5, lat_res=4, lon_offset=lon_off)
        for lon_off in lon_offsets
    ]
    frame_list_lat = [
        prep_gridded_df(ds, extents, grid_def) for grid_def in list_def_lat
    ]
    frame_list_lon = [
        prep_gridded_df(ds, extents, grid_def) for grid_def in list_def_lon
    ]
    return pd.concat(frame_list_lat + frame_list_lon)


def set_main_coords(extents=None, lon_res=5, lat_res=4):
    """Sets the base coordinates for augmentation reference."""
    if extents is None:
        extents = [-125, -65, 22, 58]
    lon_centers = np.arange(extents[0], extents[1] + lon_res, lon_res, dtype=float)
    lat_centers = np.arange(extents[2], extents[3] + lat_res, lat_res, dtype=float)
    return lon_centers, lat_centers


def get_main_coords(ds, lon_centers, lat_centers):
    """
    Returns the data array with base longitudinal coordinates only.
    Parameters: 
        - xarray dataset
        - numpy array
    Returns: 
        - xarray dataset
    """
    return (
        ds.to_dataframe()
        .reset_index()
        .merge(pd.DataFrame({"lat": lat_centers}), on="lat", how="inner")
        .merge(pd.DataFrame({"lon": lon_centers}), on="lon", how="inner")
        .set_index(["lon", "lat", "time"])
        .to_xarray()
    )


def map_transcom(ds, ds_tc):
    """
    Regrid dataset to 1-degree grid and merge TransCom regions.
    """
    # regrid the dataset to 1-degree
    df_grid = regrid(ds, lon_res=1, lat_res=1).dropna().reset_index()

    # get transcom
    df_regions = ds_tc.to_dataframe().dropna().reset_index()

    # merge, format, return
    return (
        df_grid.merge(df_regions, on=["lon", "lat"], how="inner")
        .drop(columns=["lon", "lat"])
        .dropna()
        .set_index(["time"])
    )


def to_xarray(coords, **kwargs):
    """Format data variables as xarray data array or dataset.
    
    NOTE: coords must be formatted in rows as [[lat, lon]].
    """
    return (
        pd.DataFrame({**{"lat": coords[:, 0], "lon": coords[:, 1]}, **kwargs})
        .set_index(["lon", "lat"])
        .to_xarray()
    )
