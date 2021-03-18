import numpy as np
import pandas as pd
import xarray as xr
import regionmask

from numba import njit
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances
from datetime import datetime
from dateutil import relativedelta

from stat_tools import apply_detrend
import data_utils


def count_months(d1, d2):
    """Temporal distance in months."""
    date1 = datetime.strptime(str(d1), "%Y-%m-%d")
    date2 = datetime.strptime(str(d2), "%Y-%m-%d")
    r = relativedelta.relativedelta(date2, date1)
    months = r.months + 12 * r.years
    if r.days > 0:
        months += 1
    return months


def get_offset_date_range(year, offset):
    # start_date = datetime.fromisoformat(f"{year}-01-01")
    # time_mask_xco2 = (df["time"] >= start_date) & (df["time"] < start_date + relativedelta(years=1))
    # time_mask_sif = (df["time"] >= start_date - relativedelta(months=sif_month_lag)) & (df["time"] < start_date + relativedelta(months=12-sif_month_lag))
    # df[time_mask_xco2]
    pass


def standardize_yearly_groups():
    # df.groupby("year").transform(lambda x: (x - x.mean()) / x.std())
    pass


def get_field_names(ds):
    """Returns data and estimated variance names from dataset."""
    var_name = [name for name in list(ds.keys()) if "_var" in name][0]
    data_name = var_name.replace("_var", "")
    return data_name, var_name


def preprocess_ds(ds, detrend=False, center=False, standardize=False, scale_fact=None):
    """Apply data transformations and compute surface mean and standard deviation."""
    data_name, var_name = get_field_names(ds)

    # TODO: get actual trend so it can be added back to field in prediction
    if detrend:
        ds[data_name], _ = apply_detrend(ds[data_name])

    ds["mean"] = ds[data_name].mean(dim="time")
    ds["std"] = ds[data_name].std(dim="time")

    # TODO: throw warning if both center and standardize are True
    if center:
        ds[data_name] = ds[data_name] - ds["mean"]

    if standardize:
        ds[data_name] = (ds[data_name] - ds["mean"]) / ds["std"]

    if scale_fact is not None:
        ds[data_name] = scale_fact * ds[data_name]
        ds[var_name] = scale_fact * ds[var_name]

    return ds


def land_grid(res=1, lon_lwr=-180, lon_upr=180, lat_lwr=-90, lat_upr=90):
    """Collect land locations on a regular grid as an array.

    Returns rows with entries [[lat, lon]].
    """
    # establish a fine resolution grid of 0.25 degrees for accuracy
    grid = data_utils.global_grid(
        0.25, lon_lwr=lon_lwr, lon_upr=lon_upr, lat_lwr=lat_lwr, lat_upr=lat_upr
    )
    land = regionmask.defined_regions.natural_earth.land_110
    mask = land.mask(grid["lon_centers"], grid["lat_centers"])
    # regrid to desired resolution and remove non-land areas
    df_mask = (
        data_utils.regrid(mask, res=res)
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


def distance_matrix_time(T1, T2, units="M"):
    """Computes the relative difference in months among all pairs of points given two sets of dates."""
    T1 = T1.astype(f"datetime64[{units}]")
    T2 = T2.astype(f"datetime64[{units}]")
    return np.abs(np.subtract.outer(T1, T2)).astype(float)


@njit
def get_dist_pairs(D, dist, tol=0.0):
    """Returns indices of pairs within a tolerance of the specified distance from distance matrix D."""
    if dist == 0 or tol == 0:
        pairs = np.argwhere(D == dist)
    else:
        pairs = np.argwhere((D >= dist - tol) & (D <= dist + tol))
    return pairs


@njit
def spacetime_cov_calc(data, pairs_time, pairs_space):
    """
    Computes the product of elements in x1 and x2 for each pair of spatial and temporal indices, and returns the mean of non-missing elements.
    
    Parameters:
        data: Kx4 array with columns {time_id, location_id, x1, x2}
        pairs_time: Nx2 array with columns {time_id for x1, time_id for x2}
        pairs_space: Mx2 array with columns {location_id for x1, location_id for x2}
    Returns:
        cov: mean of pairwise products
    """
    n = pairs_time.shape[0]
    m = pairs_space.shape[0]
    pairs_prod = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            point_var1 = (data[:, 0] == pairs_time[i, 0]) & (
                data[:, 1] == pairs_space[j, 0]
            )
            point_var2 = (data[:, 0] == pairs_time[i, 1]) & (
                data[:, 1] == pairs_space[j, 1]
            )
            pairs_prod[i, j] = data[point_var1][0, 2] * data[point_var2][0, 3]

    return np.nanmean(pairs_prod)


@njit
def spacetime_vario_calc(data, pairs_time, pairs_space):
    """
    Computes the squared difference for each pair of spatial and temporal indices, and returns the mean of non-missing elements.
    
    Parameters:
        data: Kx3 array with columns {time_id, location_id, x1}
        pairs_time: Nx2 array with columns {time_id for x1, time_id for x2}
        pairs_space: Mx2 array with columns {location_id for x1, location_id for x2}
    Returns:
        cov: mean of pairwise squared differences
    """
    n = pairs_time.shape[0]
    m = pairs_space.shape[0]
    pairs_var = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            point1 = (data[:, 0] == pairs_time[i, 0]) & (
                data[:, 1] == pairs_space[j, 0]
            )
            point2 = (data[:, 0] == pairs_time[i, 1]) & (
                data[:, 1] == pairs_space[j, 1]
            )
            pairs_var[i, j] = (data[point1][0, 2] - data[point2][0, 2]) ** 2

    return np.nanmean(pairs_var)


def empirical_variogram(
    df,
    vars,
    space_lags,
    time_lag=0,
    tol=None,
    cross=True,
    covariogram=False,
    standardize=False,
):
    """
    Empirical spatio-temporal (co)variogram.
    
    Params:
        df: dataframe
        vars: list of variables for which variogram will be computed
        space_lags: 1xN array of increasing spatial lags
        time_lag: integer
        tol: the width of subsequent distance intervals into which data point pairs are grouped for semivariance estimates, by default the maximum lag is divided into 15 equal windows.
        cross: indicates whether the cross covariogram will be computed
        covariogram: indicates whether the covariogram should be computed instead of the variogram
        standardize: should each data variable be locally standardized?
        
    Returns:
        vario_obj: 
    """
    assert len(vars) <= 2

    # Format data
    df["loc_id"] = df.groupby(["lat", "lon"]).ngroup()
    df["t_id"] = df.groupby(["time"]).ngroup()

    # Standardize locally or remove local mean (i.e., temporal replication)
    if standardize:
        df[vars] = df.groupby("loc_id")[vars].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    else:
        df[vars] = df.groupby("loc_id")[vars].transform(lambda x: x - x.mean())

    data_dict = dict()
    vario_obj = dict()
    for i, var in enumerate(vars):
        data_dict[var] = df[["t_id", "loc_id"] + [vars[i], vars[i]]].values
        vario_obj[var] = np.zeros_like(space_lags)

    if cross:
        assert len(vars) > 1
        data_dict[f"{vars[0]}:{vars[1]}"] = df[["t_id", "loc_id"] + vars].values
        vario_obj[f"{vars[0]}:{vars[1]}"] = np.zeros_like(space_lags)

    # Establish common spatiotemporal domain (may lead to missing data values)
    temporal_domain = np.unique(df["time"].values)
    spatial_domain = np.unique(df[["lat", "lon"]].values, axis=0)

    # Precompute distances
    dist_time = distance_matrix_time(temporal_domain, temporal_domain)
    dist_space = distance_matrix(spatial_domain, spatial_domain, fast_dist=True)

    # Get temporal pairs
    pairs_time = get_dist_pairs(dist_time, time_lag)

    # Iterate over space_lags, get spatial pairs, compute covariances
    if tol is None:
        tol = space_lags[-1] / 15

    # TODO: consider reformatting so max space lag is within max dist
    if covariogram:
        # TODO: make this piece a sperate function
        for var in data_dict.keys():
            for i, lag in enumerate(space_lags):
                pairs_space = get_dist_pairs(dist_space, lag, tol=tol)
                assert (
                    pairs_space.shape[0] > 0
                ), "Error: Distance too specific to obtain point pairs. Try increasing spatial tolerance."
                vario_obj[var][i] = spacetime_cov_calc(
                    data_dict[var], pairs_time, pairs_space
                )
    else:
        for var in data_dict.keys():
            for i, lag in enumerate(space_lags):
                pairs_space = get_dist_pairs(dist_space, lag, tol=tol)
                assert (
                    pairs_space.shape[0] > 0
                ), "Error: Distance too specific to obtain point pairs. Try increasing spatial tolerance."
                vario_obj[var][i] = spacetime_vario_calc(
                    data_dict[var], pairs_time, pairs_space
                )

    return vario_obj


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


def pre_post_diag(u, A, v=None):
    """Returns the matrix product: diag(u) A diag(v).

    params:
        - v, u: vector(s) passed to np.diag()
        - A: matrix
    """
    if v is None:
        v = u
    return np.matmul(np.diag(u), np.matmul(A, np.diag(v)))

