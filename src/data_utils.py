# Utilities for reading, writing, and formatting data
import numpy as np
import pandas as pd
import xarray

# ---------------------------------------------------
# Reading
# ---------------------------------------------------
def prep_sif(ds):
    """Preprocess an OCO-2 SIF Lite file"""
    
    # Drop unused variables
    variable_list = ["Daily_SIF_740nm", "SIF_Uncertainty_740nm", "Quality_Flag", "Longitude", "Latitude", "Delta_Time"]
    ds = ds[variable_list]
    
    # Apply quality filters
    ds["SIF_plus_3sig"] = ds.Daily_SIF_740nm + 3*ds.SIF_Uncertainty_740nm
    ds = ds.where(ds.Quality_Flag != 2, drop=True)
    ds = ds.where(ds.SIF_plus_3sig > 0, drop=True)

    # Format dataset
    return xarray.Dataset(
        {
            "sif": (["time"], ds.Daily_SIF_740nm),
        },
        coords={
            "lon": (["time"], ds.Longitude),
            "lat": (["time"], ds.Latitude),
            "time": ds.Delta_Time.values
        }
    )

def prep_xco2(ds):
    """Preprocess an OCO-2 FP Lite file"""
    
    # Drop unused variables
    variable_list = ["xco2", "xco2_quality_flag", "longitude", "latitude", "time"]
    ds = ds[variable_list]
    
    # Apply quality filters
    ds = ds.where(ds.xco2_quality_flag == 0, drop=True)

    # Format dataset
    return xarray.Dataset(
        {
            "xco2": (["time"], ds.xco2),
        },
        coords={
            "lon": (["time"], ds.longitude),
            "lat": (["time"], ds.latitude),
            "time": ds.time.values
        }
    )

def read_transcom(path):
    """
    Read 1-degree TransCom 3 region map into xarray dataset.
    """
    ds = xarray.open_dataset(path)
    ds = ds.where(ds.region < 12, drop=True)
    ds = ds.where(ds.region != 0, drop=True)
    
    return ds


# ---------------------------------------------------
# Formatting
# ---------------------------------------------------
def regrid(ds, res=1):
    """
    Convert dataset to dataframe and assign coordinates using a regular grid.
    """
    df = ds.to_dataframe()
        
    # Establish grid
    lon_bins = np.arange(-180, 180+res, res)
    lat_bins = np.arange(-90, 90+res, res)
    lon_centers = (lon_bins[1:] + lon_bins[:-1]) / 2
    lat_centers = (lat_bins[1:] + lat_bins[:-1]) / 2

    # Overwrite lon-lat values with grid values
    df["lon"] = pd.cut(df.lon, lon_bins, labels=lon_centers).astype(float)
    df["lat"] = pd.cut(df.lat, lat_bins, labels=lat_centers).astype(float)
    
    return df

def map_transcom(ds, ds_tc):
    """
    Regrid dataset to 1-degree grid and merge TransCom regions.
    """
    # Regrid the dataset to 1-degree
    df_grid = regrid(ds, res=1).dropna().reset_index()
    
    # Get transcom
    df_regions = ds_tc.to_dataframe().dropna().reset_index()
    
    # Merge, format, return
    return (df_grid
            .merge(df_regions, on=["lon", "lat"], how="inner")
            .drop(columns=["lon", "lat"])
            .dropna()
            .set_index(["time"])
           )