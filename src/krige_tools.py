import numpy as np
import pandas as pd
import xarray as xr

from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances

def preprocess_da(da, detrend=False, standardize=False):
    """Preprocess a data array for cokriging."""
    return da


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
