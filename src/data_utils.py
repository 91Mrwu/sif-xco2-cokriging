# Utilities for reading, writing, and formatting data
import numpy as np
import pandas as pd
import xarray

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
    return xarray.Dataset(
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
    xco2_uncertainty is defined as "the posterior uncertainty in XCO2 calculated by the L2 algorithm, in ppm. This is generally 30-50% smaller than the true retrieval uncertainty." Doubling this value yeilds a conservative estimate of the variance of the measurement error which will be added to the diagonals of the covariance matrix.
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
    return xarray.Dataset(
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
    ds = xarray.open_dataset(path)
    ds = ds.where(ds.region < 12, drop=True)
    ds = ds.where(ds.region != 0, drop=True)

    return ds


## Formatting
def regrid(ds, res=1):
    """
    Convert dataset to dataframe and assign coordinates using a regular grid.
    """
    df = ds.to_dataframe()

    # establish grid
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)
    lon_centers = (lon_bins[1:] + lon_bins[:-1]) / 2
    lat_centers = (lat_bins[1:] + lat_bins[:-1]) / 2

    # overwrite lon-lat values with grid values
    df["lon"] = pd.cut(df.lon, lon_bins, labels=lon_centers).astype(float)
    df["lat"] = pd.cut(df.lat, lat_bins, labels=lat_centers).astype(float)

    return df


def map_transcom(ds, ds_tc):
    """
    Regrid dataset to 1-degree grid and merge TransCom regions.
    """
    # regrid the dataset to 1-degree
    df_grid = regrid(ds, res=1).dropna().reset_index()

    # get transcom
    df_regions = ds_tc.to_dataframe().dropna().reset_index()

    # merge, format, return
    return (
        df_grid.merge(df_regions, on=["lon", "lat"], how="inner")
        .drop(columns=["lon", "lat"])
        .dropna()
        .set_index(["time"])
    )
