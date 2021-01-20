import numpy as np
import pandas as pd
import xarray as xr

from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances

from stat_tools import apply_detrend
import krige_tools

"""
TODO: SIF values are on the order of 0-8, where XCO2 values are in the hundreds. Since XCO2 is easiest to preform a unit conversion on, apply a scale factor of 1/1000 to convert units to parts per billion.
"""


def get_field_names(ds):
    """Returns data and estimated variance names from dataset."""
    var_name = [name for name in list(ds.keys()) if "_var" in name][0]
    data_name = var_name.replace("_var", "")
    return data_name, var_name


def preprocess_ds(ds, detrend=False, standardize=False, scale_fact=None):
    """Apply data transformations and compute surface mean and standard deviation."""
    data_name, var_name = krige_tools.get_field_names(ds)

    if detrend:
        ds[data_name], _ = apply_detrend(ds[data_name])

    ds["mean"] = ds[data_name].mean(dim="time")
    ds["std"] = ds[data_name].std(dim="time")

    if standardize:
        ds[data_name] = (ds[data_name] - ds["mean"]) / ds["std"]

    if scale_fact is not None:
        ds[data_name] = scale_fact * ds[data_name]
        ds[var_name] = scale_fact * ds[var_name]

    return ds


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
