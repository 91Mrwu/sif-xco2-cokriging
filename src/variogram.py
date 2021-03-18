import numpy as np
import pandas as pd
from numba import njit, prange

from krige_tools import distance_matrix


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


@njit(parallel=True)
def apply_vario_calc(space_lags, dist_space, tol, data, pairs_time, covariogram):
    """For a fixed temporal lag, compute vario calc at all spatial lags in parallel."""
    v = np.zeros_like(space_lags)

    for h in prange(len(v)):
        pairs_space = get_dist_pairs(dist_space, space_lags[h], tol=tol)
        # TODO: determine how to check this condition in parallel setting
        # if pairs_space.shape[0] == 0:
        #     raise ValueError(
        #         "Distance too specific to obtain point pairs. Try increasing spatial tolerance."
        #     )
        if covariogram:
            v[h] = spacetime_cov_calc(data, pairs_time, pairs_space)
        else:
            v[h] = spacetime_vario_calc(data, pairs_time, pairs_space)

    return v


def empirical_variogram(
    df,
    vars,
    space_lags,
    tol=None,
    time_lag=0,
    cross=True,
    covariogram=False,
    standardize=False,
):
    """
    Empirical spatio-temporal (co)variogram.
    
    Parameters:
        df: dataframe with columns {"lat", "lon", "time", variable1, variable2}
        vars: list of variables for which variogram will be computed
        space_lags: 1xN array of increasing spatial lags
        tol: radius of the spatial neighborhood into which data point pairs are grouped for semivariance estimates, by default the maximum lag is divided by 15
        time_lag: integer
        cross: indicates whether the cross (co)variogram will be computed
        covariogram: indicates whether the covariogram should be computed instead of the variogram
        standardize: should each data variable be locally standardized?
        
    Returns:
        df_vario: dataframe containing the spatial lags and corresponding (co)variogram values 
    """
    assert len(vars) <= 2
    if tol is None:
        tol = space_lags[-1] / 15

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

    df_vario = pd.DataFrame({"lag": space_lags})
    data_dict = dict()
    for i, var in enumerate(vars):
        data_dict[var] = df[["t_id", "loc_id"] + [vars[i], vars[i]]].values
    if cross:
        assert len(vars) > 1
        data_dict[f"{vars[0]}:{vars[1]}"] = df[["t_id", "loc_id"] + vars].values

    # Establish common spatiotemporal domain (may lead to missing data values)
    temporal_domain = np.unique(df["time"].values)
    spatial_domain = np.unique(df[["lat", "lon"]].values, axis=0)

    # Precompute distances
    dist_time = distance_matrix_time(temporal_domain, temporal_domain)
    dist_space = distance_matrix(spatial_domain, spatial_domain, fast_dist=True)
    assert time_lag <= dist_time.max()
    assert space_lags[-1] <= dist_space.max()

    # Get temporal pairs
    pairs_time = get_dist_pairs(dist_time, time_lag)

    # Compute variograms
    for var in data_dict.keys():
        df_vario[var] = apply_vario_calc(
            space_lags, dist_space, tol, data_dict[var], pairs_time, covariogram
        )

    return df_vario
