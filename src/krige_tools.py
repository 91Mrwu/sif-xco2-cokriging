import warnings

from numba import njit
import numpy as np
import pandas as pd
import xarray as xr
import regionmask

from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import data_utils
from stat_tools import simple_linear_regression


def get_field_names(ds):
    """Returns data and estimated variance names from dataset."""
    var_name = [name for name in list(ds.keys()) if "_var" in name][0]
    data_name = var_name.replace("_var", "")
    return data_name, var_name


def remove_linear_trend(da):
    """Computes the monthly average of all spatial locations, and removes the trend fit by a linear model."""
    x = da.mean(dim=["lat", "lon"])
    trend = simple_linear_regression(x.values)
    da_trend = xr.DataArray(trend, dims=["time"], coords={"time": da.time})
    return da - da_trend


def fit_ols(ds, data_name):
    """Estimate the mean surface using ordinary least squares."""
    df = ds[data_name].to_dataframe().dropna().drop(columns=["time"]).reset_index()
    if df.shape[0] == 0:
        # no data
        return ds[data_name] * np.nan
    else:
        if data_name == "sif":
            X = df["lon"].values.reshape(-1, 1)
        else:
            X = df[["lon", "lat"]]
        model = LinearRegression().fit(X, df.iloc[:, -1])
        df = df.iloc[:, :-1]
        df["ols_mean"] = model.predict(X)
        return (
            df.set_index(["lon", "lat"])
            .to_xarray()
            .assign_coords(coords={"time": ds[data_name].time})["ols_mean"]
        )


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


# def match_data_locations(field_1, field_2):
#     """Only keep data at shared locations"""
#     df_1 = pd.DataFrame(
#         {
#             "lat": field_1.coords[:, 0],
#             "lon": field_1.coords[:, 1],
#             "values": field_1.values,
#         }
#     )
#     df_2 = pd.DataFrame(
#         {
#             "lat": field_2.coords[:, 0],
#             "lon": field_2.coords[:, 1],
#             "values": field_2.values,
#         }
#     )
#     df = pd.merge(df_1, df_2, on=["lat", "lon"], suffixes=("_1", "_2"))
#     return df.values_1, df.values_2


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

