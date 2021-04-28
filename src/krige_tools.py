from datetime import datetime
from dateutil.relativedelta import relativedelta

from numba import njit
import numpy as np
import pandas as pd
import xarray as xr
import regionmask

from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances

from stat_tools import apply_detrend
import data_utils


def get_year_window(timestamp):
    """Given the month to center on, return the first and last month in the window as a list."""
    center_time = datetime.strptime(timestamp, "%Y-%m-%d")
    window = [
        center_time - relativedelta(months=5),
        center_time + relativedelta(months=6),
    ]
    return tuple([w.strftime("%Y-%m-%d") for w in window])


def get_date_range_offset(df, vars, year, offset):
    """Select a year of data for both variables, with the second variable lagged by the offset."""
    df["time"] = pd.to_datetime(df["time"])
    start_date = datetime.fromisoformat(f"{year}-01-01")
    mask = (df["time"] >= start_date) & (
        df["time"] < start_date + relativedelta(years=1)
    )
    mask_offset = (df["time"] >= start_date - relativedelta(months=offset)) & (
        df["time"] < start_date + relativedelta(months=12 - offset)
    )
    df_var1 = df[mask].drop(vars[1], axis=1)
    df_var2 = df[mask_offset].drop(vars[0], axis=1)
    df_offset = pd.merge(df_var1, df_var2, how="outer", on=["lat", "lon", "time"])
    return df_offset


def get_field_names(ds):
    """Returns data and estimated variance names from dataset."""
    var_name = [name for name in list(ds.keys()) if "_var" in name][0]
    data_name = var_name.replace("_var", "")
    return data_name, var_name


def preprocess_ds(ds, timestamp, full_detrend=False, standardize_window=False):
    """Apply data transformations and compute surface mean and standard deviation."""
    data_name, var_name = get_field_names(ds)

    # TODO: get actual trend so it can be added back to field in prediction
    if full_detrend:
        ds[data_name], _ = apply_detrend(ds[data_name])

    # Subset dataset to year centered on timestamp
    window = get_year_window(timestamp)
    ds_window = ds.sel(time=slice(*window))

    ds_window["temporal_mean"] = ds_window[data_name].mean(dim="time")
    ds_window["temporal_std"] = ds_window[data_name].std(dim="time")

    if standardize_window:
        ds_window[data_name] = (
            ds_window[data_name] - ds_window["temporal_mean"]
        ) / ds_window["temporal_std"]

    # Temporally-indexed spatial means may not be stationary in time
    ds_window["spatial_mean"] = ds_window[data_name].mean(dim=["lon", "lat"])

    return ds_window


def land_grid(lon_res=1, lat_res=1, lon_lwr=-180, lon_upr=180, lat_lwr=-90, lat_upr=90):
    """Collect land locations on a regular grid as an array.

    Returns rows with entries [[lat, lon]].
    """
    # establish a fine resolution grid of 0.25 degrees for accuracy
    grid = data_utils.global_grid(
        0.25, 0.25, lon_lwr=lon_lwr, lon_upr=lon_upr, lat_lwr=lat_lwr, lat_upr=lat_upr
    )
    land = regionmask.defined_regions.natural_earth.land_110
    mask = land.mask(grid["lon_centers"], grid["lat_centers"])
    # regrid to desired resolution and remove non-land areas
    df_mask = (
        data_utils.regrid(
            mask,
            lon_res=lon_res,
            lat_res=lat_res,
            lon_lwr=lon_lwr,
            lon_upr=lon_upr,
            lat_lwr=lat_lwr,
            lat_upr=lat_upr,
        )
        .dropna(subset=["region"])
        .groupby(["lon", "lat"])
        .mean()
        .reset_index()
    )
    return df_mask[["lat", "lon"]].values


def expand_grid(*args):
    """
    Returns an array of all combinations of elements in the supplied vectors.
    """
    return np.array(np.meshgrid(*args)).T.reshape(-1, len(args))


def distance_matrix(X1, X2, units="km", fast_dist=False):
    """
    Computes the geodesic (or great circle if fast_dist=True) distance among all pairs of points given two sets of coordinates.
    Wrapper for scipy.spatial.distance.cdist using geopy.distance.geodesic as a the metric.

    NOTE: 
    - points should be formatted in rows as [lat, lon]
    - if fast_dist=True, units are kilometers regardless of specification
    """
    # enforce 2d array in case of single point
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    if fast_dist:
        # great circle distances in kilometers
        EARTH_RADIUS = 6371  # radius in kilometers
        X1_r = np.radians(X1)
        X2_r = np.radians(X2)
        return haversine_distances(X1_r, X2_r) * EARTH_RADIUS
    else:
        # geodesic distances in specified units
        return cdist(X1, X2, lambda s_i, s_j: getattr(geodesic(s_i, s_j), units))


def match_data_locations(field_1, field_2):
    """Only keep data at shared locations"""
    df_1 = pd.DataFrame(
        {
            "lat": field_1.coords[:, 0],
            "lon": field_1.coords[:, 1],
            "values": field_1.values,
        }
    )
    df_2 = pd.DataFrame(
        {
            "lat": field_2.coords[:, 0],
            "lon": field_2.coords[:, 1],
            "values": field_2.values,
        }
    )
    df = pd.merge(df_1, df_2, on=["lat", "lon"], suffixes=("_1", "_2"))
    return df.values_1, df.values_2


# TODO: test whether numba is actually faster here using toy arrays
# @njit
def pre_post_diag(u, A, v=None):
    """Returns the matrix product: diag(u) A diag(v).

    params:
        - v, u: vector(s) passed to np.diag()
        - A: matrix
    """
    if v is None:
        v = u
    return np.matmul(np.diag(u), np.matmul(A, np.diag(v)))
    # return np.diag(u) @ A @ np.diag(v)  # matmul doesn't play with numba

