import numpy as np
from pandas import DataFrame
from xarray import DataArray

from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances
from sklearn.linear_model import LinearRegression

from stat_tools import standardize, simple_linear_regression


def fit_linear_trend(da):
    """Computes the monthly average of all spatial locations, and removes the trend fit by a linear model."""
    x = da.mean(dim=["lat", "lon"])
    trend = simple_linear_regression(x.values)
    return DataArray(trend, dims=["time"], coords={"time": da.time})


def fit_ols(ds, data_name, covar_names: list):
    """Fit and predict the mean surface using ordinary least squares with standarized covariates."""
    df = (
        ds.to_dataframe()
        .drop(columns=["time", f"{data_name}_var"])
        .dropna(subset=[data_name])
        .reset_index()
    )
    if df.shape[0] == 0:  # no data
        return ds[data_name] * np.nan
    covariates = df[covar_names].apply(lambda x: standardize(x), axis=0)
    model = LinearRegression().fit(covariates, df[data_name])
    df = df[["lon", "lat"]]
    df["ols_mean"] = model.predict(covariates)
    ds_pred = (
        df.set_index(["lon", "lat"])
        .to_xarray()
        .assign_coords(coords={"time": ds[data_name].time})
    )
    return ds_pred["ols_mean"]


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


def get_group_ids(group: DataFrame):
    """Returns the group ids as a tuple (i, j)."""
    i = group.index.get_level_values("i")[0]
    j = group.index.get_level_values("j")[0]
    return i, j
