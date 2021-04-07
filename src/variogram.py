import warnings

from numba import njit, prange
import numpy as np
import pandas as pd
import scipy.special as sps
from scipy.optimize import curve_fit, minimize

from krige_tools import distance_matrix

# TODO: establish a variogram class


def crop_lag_vec(lags, dist, frac=0.65):
    """Crop lag vector from minimum distance to a fraction of the maximum distance and pad with a leading zero."""
    min_dist = dist[dist > 0.0].min()
    max_dist = frac * dist.max()
    lags = lags[(lags >= min_dist) & (lags <= max_dist)]
    return np.hstack([[0.0], lags])


def distance_matrix_time(T1, T2, units="M"):
    """Computes the relative difference in months among all pairs of points given two sets of dates."""
    T1 = T1.astype(f"datetime64[{units}]")
    T2 = T2.astype(f"datetime64[{units}]")
    return np.abs(np.subtract.outer(T1, T2)).astype(float)


@njit
def get_dist_pairs(D, dist, tol=0.0):
    """Returns indices of pairs within a tolerance of the specified distance from distance matrix D."""
    # NOTE: to get neg. dists, consider pairs[pairs[:, 0] < pairs[:, 1]] from "directional setting"
    if dist == 0 or tol == 0:
        pairs = np.argwhere(D == dist)
    else:
        pairs = np.argwhere((D >= dist - tol) & (D <= dist + tol))
    return pairs


@njit
def directional_pairs(D, dist):
    """Returns backward pairs. For use with temporal distance matrix D."""
    pairs = get_dist_pairs(D, dist)
    return pairs[pairs[:, 0] >= pairs[:, 1]]


@njit
def spacetime_vario_calc(data, pairs_time, pairs_space):
    """
    Computes the squared difference for each pair of spatial and temporal indices, and returns the mean of non-missing elements.
    
    Parameters:
        data: Kx3 array with columns {time_id, location_id, values}
        pairs_time: Nx2 array with columns {time_id, time_id}
        pairs_space: Mx2 array with columns {location_id, location_id} 
    Returns:
        vario: mean of pairwise squared differences
        count: number on non-missing pairs included in calculation
    """
    n = pairs_time.shape[0]
    m = pairs_space.shape[0]
    pairs_var = np.nan * np.zeros((n, m))

    if n == 0 or m == 0:
        return np.nan, 0.0

    for i in range(n):  # temporal ids
        for j in range(m):  # spatial ids
            point1 = (data[:, 0] == pairs_time[i, 0]) & (
                data[:, 1] == pairs_space[j, 0]
            )
            point2 = (data[:, 0] == pairs_time[i, 1]) & (
                data[:, 1] == pairs_space[j, 1]
            )
            diff = data[point1] - data[point2]
            if diff.size:  # array is not empty, data available at matched points
                pairs_var[i, j] = (diff[0, 2]) ** 2
            else:
                pairs_var[i, j] = np.nan

    return np.nanmean(pairs_var), np.count_nonzero(~np.isnan(pairs_var))


@njit
def spacetime_cov_calc(d1, d2, pairs_time, pairs_space):
    """
    Computes the product of elements in x1 and x2 for each pair of spatial and temporal indices, and returns the mean of non-missing elements.
    
    Parameters:
        d1, d2: Kx3 arrays with columns {time_id, location_id, values}
        pairs_time: Nx2 array with columns {time_id for x1, time_id for x2}
        pairs_space: Mx2 array with columns {location_id for x1, location_id for x2}
    Returns:
        cov: mean of pairwise products
        count: number on non-missing pairs included in calculation
    """
    n = pairs_time.shape[0]
    m = pairs_space.shape[0]
    pairs_prod = np.nan * np.zeros((n, m))
    if n == 0 or m == 0:
        return np.nan, 0.0

    for i in range(n):  # temporal ids
        for j in range(m):  # spatial ids
            point_var1 = (d1[:, 0] == pairs_time[i, 0]) & (
                d1[:, 1] == pairs_space[j, 0]
            )
            point_var2 = (d2[:, 0] == pairs_time[i, 1]) & (
                d2[:, 1] == pairs_space[j, 1]
            )
            prod = d1[point_var1] * d2[point_var2]
            if prod.size:  # array is not empty, data available at matched points
                pairs_prod[i, j] = prod[0, 2]
            else:
                pairs_prod[i, j] = np.nan

    return np.nanmean(pairs_prod), np.count_nonzero(~np.isnan(pairs_prod))


@njit(parallel=True)
def apply_vario_calc(space_lags, dist_space, tol, pairs_time, data):
    """For a fixed temporal lag, compute vario calc at all spatial lags in parallel."""
    v = np.zeros_like(space_lags)
    counts = np.zeros_like(space_lags)

    for h in prange(len(v)):  # pylint: disable=not-an-iterable
        pairs_space = get_dist_pairs(dist_space, space_lags[h], tol=tol)
        if pairs_space.shape[0] == 0:
            print("Degenerate.")
        v[h], counts[h] = spacetime_vario_calc(data, pairs_time, pairs_space)

    return v, counts


@njit(parallel=True)
def apply_xcov_calc(space_lags, dist_space, tol, pairs_time, data1, data2):
    """For a fixed temporal lag, compute cross covariance at all spatial lags in parallel."""
    v = np.zeros_like(space_lags)
    counts = np.zeros_like(space_lags)

    for h in prange(len(v)):  # pylint: disable=not-an-iterable
        pairs_space = get_dist_pairs(dist_space, space_lags[h], tol=tol)
        if pairs_space.shape[0] == 0:
            print("Degenerate.")
        v[h], counts[h] = spacetime_cov_calc(data1, data2, pairs_time, pairs_space)

    return v, counts


def empirical_variogram(
    df,
    name,
    space_lags,
    tol,
    time_lag,
    crop_lags=0.65,
    covariogram=True,
    normalize=False,
):
    """Basic function to compute a variogram from a dataframe."""
    # Establish space-time domain
    times = np.unique(df["time"].values)
    coords = np.unique(df[["lat", "lon"]].values, axis=0)

    # Precompute distances
    dist_time = distance_matrix_time(times, times)
    dist_space = distance_matrix(coords, coords, fast_dist=True)
    if crop_lags:
        space_lags = crop_lag_vec(space_lags, dist_space, frac=crop_lags)

    assert space_lags[-1] <= dist_space.max()
    assert time_lag <= dist_time.max()

    # Get temporal pairs
    pairs_time = get_dist_pairs(dist_time, time_lag)

    # Format data and variogram dataframe
    data = df[["t_id", "loc_id"] + [name]].values
    df_vgm = pd.DataFrame({"lag": space_lags})

    # Compute variogram
    if covariogram:
        df_vgm[name], df_vgm["counts"] = apply_xcov_calc(
            space_lags, dist_space, tol, pairs_time, data, data
        )
        if normalize:
            df_vgm[name] = df_vgm[name] / df_vgm[name][0]
    else:
        df_vgm[name], df_vgm["counts"] = apply_vario_calc(
            space_lags, dist_space, tol, pairs_time, data
        )
    if (df_vgm["counts"] < 30).any():
        warnings.warn(
            f"WARNING: Fewer than 30 pairs used for at least one bin in variogram calculation."
        )

    return df_vgm


def empirical_cross_cov(
    data_dict, space_lags, tol, time_lag, crop_lags=0.65, normalize=False
):
    """Basic function to compute a cross covariogram from a pair of dataframes stored in dict."""
    names = list(data_dict.keys())
    # Establish space-time domains
    times = [np.unique(data_dict[name]["time"].values) for name in names]
    coords = [
        np.unique(data_dict[name][["lat", "lon"]].values, axis=0) for name in names
    ]

    # Precompute distances
    dist_time = distance_matrix_time(times[0], times[1])
    dist_space = distance_matrix(coords[0], coords[1], fast_dist=True)
    if crop_lags:
        space_lags = crop_lag_vec(space_lags, dist_space, frac=crop_lags)

    assert space_lags[-1] <= dist_space.max()
    assert time_lag <= dist_time.max()

    # Get directional temporal pairs (don't assume symmetry)
    pairs_time = directional_pairs(dist_time, time_lag)

    # Format data and variogram dataframe
    data = [data_dict[name][["t_id", "loc_id"] + [name]].values for name in names]
    df_cov = pd.DataFrame({"lag": space_lags})

    # Compute cross-covariograms
    cross_name = f"{names[0]}:{names[1]}"
    df_cov[cross_name], df_cov["counts"] = apply_xcov_calc(
        space_lags, dist_space, tol, pairs_time, data[0], data[1]
    )
    if normalize:
        df_cov[cross_name] = df_cov[cross_name] / np.abs(df_cov[cross_name][0])
    if (df_cov["counts"] < 30).any():
        warnings.warn(
            f"WARNING: Fewer than 30 pairs used for at least one bin in covariogram calculation."
        )

    return df_cov


# TODO: add ability to `freeze` parameters
def matern_correlation(xdata, len_scale):
    nu = 1.5
    xdata_ = xdata[xdata > 0.0] / len_scale
    corr = np.ones_like(xdata)
    corr[xdata > 0.0] = np.exp(
        (1.0 - nu) * np.log(2)
        - sps.gammaln(nu)
        + nu * np.log(np.sqrt(2.0 * nu) * xdata_)
    ) * sps.kv(nu, np.sqrt(2.0 * nu) * xdata_)

    corr[np.logical_not(np.isfinite(corr))] = 0.0
    # normalized Matern is positive
    corr = np.maximum(corr, 0.0)
    return corr


def matern_vario(xdata, sigma, len_scale, nugget):
    return sigma ** 2 * (1 - matern_correlation(xdata, len_scale)) + nugget


def matern_cov(xdata, sigma, len_scale, nugget):
    cov = sigma ** 2 * matern_correlation(xdata, len_scale)
    cov[xdata == 0] += nugget
    return cov


def matern_cross_cov(xdata, sigmas, len_scale, rho):
    return rho * sigmas[0] * sigmas[1] * matern_correlation(xdata, len_scale)


def matern_cross_corr(xdata, len_scale, rho):
    return rho * matern_correlation(xdata, len_scale)


def weighted_least_squares(ydata, yfit, bin_counts):
    zeros = np.argwhere(yfit == 0.0)
    non_zero = np.argwhere(yfit != 0.0)
    wls = np.zeros_like(yfit)
    wls[zeros] = bin_counts[zeros] * ydata[zeros] ** 2  # NOTE: is this wrong to do?
    wls[non_zero] = (
        bin_counts[non_zero]
        * ((ydata[non_zero] - yfit[non_zero]) / yfit[non_zero]) ** 2
    )
    return np.sum(wls)


def wls_cost(params, xdata, ydata, bin_counts, sigmas, covariogram):
    if sigmas is not None:
        yfit = matern_cross_cov(xdata, sigmas, *params)
    elif covariogram:
        yfit = matern_cov(xdata, *params)
    else:
        yfit = matern_vario(xdata, *params)
    return weighted_least_squares(ydata, yfit, bin_counts)


def wls_cost_norm(params, xdata, ydata, bin_counts, cross):
    if cross:
        yfit = matern_cross_corr(xdata, *params)
    else:
        yfit = matern_correlation(xdata, *params)
    return weighted_least_squares(ydata, yfit, bin_counts)


def fit_variogram(xdata, ydata, initial_guess, cross=False):
    """Fit covariance parameters to empirical variogram by non-linear least squares."""
    kwargs = {"loss": "soft_l1", "verbose": 1}
    if cross:
        # Fit using covariogram
        bounds = ([0.01, 0.01, 0.1, -1.0], [10.0, 10.0, 1e4, 1.0])  # (lwr, upr)
        params, _ = curve_fit(
            matern_cross_cov, xdata, ydata, p0=initial_guess, bounds=bounds, **kwargs
        )
        fit = matern_cross_cov(xdata, *params)
    else:
        # Fit using variogram
        bounds = ([0.01, 0.1, 0.0], [10.0, 1e4, 10.0])  # (lwr, upr)
        params, _ = curve_fit(
            matern_vario, xdata, ydata, p0=initial_guess, bounds=bounds, **kwargs
        )
        fit = matern_vario(xdata, *params)
    return params, fit


def fit_variogram_wls(
    xdata,
    ydata,
    bin_counts,
    initial_guess,
    sigmas=None,
    covariogram=True,
    cross=False,
    normalized=False,
):
    """
    Fit covariance parameters to empirical (co)variogram by weighted least squares (Cressie, 1985).
    
    Parameters:
        xdata: pd.series giving the spatial lags
        ydata: pd.series giving the empirical variogram values to be fitted
        bin_counts: pd.series indicating the number of spatio-temporal point pairs used to calculate each empirical value
        initial_guess: list of parameter starting values given as one of [sigma, len_scale, nugget] or [len_scale, rho]
        sigmas: list of standard deviations if fitting a cross covariance
        cross: cross dependence?
        normalized: correlation (True), covariance (False)
    Returns:
        params: list of parameter values
        fit: the theoretical (co)variogram fit
    """
    pred = np.linspace(xdata.min(), xdata.max(), 100)
    if covariogram:
        if cross:
            assert len(initial_guess) == 2
            bounds = [(xdata[xdata > 0].min(), 1.5e4), (-1.5, -0.1)]
            if normalized:
                # Cross correlation, fit cross correlogram
                optim_result = minimize(
                    wls_cost_norm,
                    initial_guess,
                    args=(xdata.values, ydata.values, bin_counts.values, True),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                fit = matern_cross_corr(pred, *optim_result.x)
            else:
                # Cross covariance, fit cross covariogram
                assert sigmas is not None
                optim_result = minimize(
                    wls_cost,
                    initial_guess,
                    args=(
                        xdata.values,
                        ydata.values,
                        bin_counts.values,
                        sigmas,
                        covariogram,
                    ),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                fit = matern_cross_cov(pred, sigmas, *optim_result.x)
        else:
            if normalized:
                # Univariate correlation, fit correlogram
                assert len(initial_guess) == 1
                bounds = [(xdata[xdata > 0].min(), 1.5e4)]
                optim_result = minimize(
                    wls_cost_norm,
                    initial_guess,
                    args=(xdata.values, ydata.values, bin_counts.values, False),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                fit = matern_correlation(pred, *optim_result.x)
            else:
                # Univariate covariance, fit covariogram
                assert len(initial_guess) == 3
                bounds = [(0.01, 4.0), (xdata[xdata > 0].min(), 1.5e4), (0.0, 1.0)]
                optim_result = minimize(
                    wls_cost,
                    initial_guess,
                    args=(
                        xdata.values,
                        ydata.values,
                        bin_counts.values,
                        None,
                        covariogram,
                    ),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                # fit = matern_vario(xdata, *optim_result.x)
                fit = matern_cov(pred, *optim_result.x)
    else:
        # Univariate covariance, fit variogram
        assert len(initial_guess) == 3
        bounds = [(0.01, 4.0), (xdata[xdata > 0].min(), 1.5e4), (0.0, 1.0)]
        optim_result = minimize(
            wls_cost,
            initial_guess,
            args=(xdata.values, ydata.values, bin_counts.values, None, covariogram),
            method="L-BFGS-B",
            bounds=bounds,
        )
        fit = matern_vario(pred, *optim_result.x)

    if optim_result.success == False:
        warnings.warn("ERROR: optimization did not converge.")

    return optim_result.x, pd.DataFrame({"lag": pred, "wls_fit": fit})


def variogram_analysis(
    mf,
    space_lags,
    tol,
    cov_guess,
    cross_guess,
    crop_lags=0.65,
    standardize=False,
    covariograms=True,
    normalize_cov=False,
):
    """
    Compute the empirical spatio-temporal variograms from a multi-field object and find the weighted least squares fit.

    Parameters:
        mf: multi-field object
        space_lags: 1xN array of increasing spatial lags
        tol: radius of the spatial neighborhood into which data point pairs are grouped for semivariance estimates; note that this can be seen as a rolling window so depending on the size, some pairs may be repeated in multiple bins
        crop_lags: should spatial lag vector be trimmed to a fraction of the maximum distance, and formatted such that the first non-zero element is at least the minimum distance?
        standardize: should each data variable be locally standardized?
        cov_guess: covariance params initial guess for WLS fit; list [sigma, len_scale, nugget]
        cross_guess: cross-cov parmas initial guess for WLS fit; list [len_scale, rho]

    Returns:
        variograms: dictionary containing variogram and cross-covariogram dataframes 
        params_fit: dictionary of parameter fits for each variogram and cross-covariogram
    """
    # Format data
    data_dict = {
        mf.field_1.data_name: mf.field_1.get_spacetime_df(),
        mf.field_2.data_name: mf.field_2.get_spacetime_df(),
    }
    names = list(data_dict.keys())
    time_lag = np.abs(mf.timedelta)

    # Standardize locally or remove local mean (i.e., temporal replication)
    for name in names:
        if standardize:
            data_dict[name][name] = (
                data_dict[name]
                .groupby("loc_id")[name]
                .transform(lambda x: (x - x.mean()) / x.std())
            )
        else:
            data_dict[name][name] = (
                data_dict[name]
                .groupby("loc_id")[name]
                .transform(lambda x: x - x.mean())
            )

    # Compute and fit variograms and cross-covariogram
    variograms = dict()
    params_fit = dict()
    sigmas = list()
    for name in names:
        # NOTE: no temporal lag in variograms
        variograms[name] = empirical_variogram(
            data_dict[name],
            name,
            space_lags,
            tol,
            0,
            crop_lags=crop_lags,
            covariogram=covariograms,
            normalize=normalize_cov,
        )
        params_fit[name], df_fit = fit_variogram_wls(
            variograms[name]["lag"],
            variograms[name][name],
            variograms[name]["counts"],
            cov_guess,
            covariogram=covariograms,
            normalized=normalize_cov,
        )
        variograms[name] = pd.merge(variograms[name], df_fit, on="lag", how="outer")
        sigmas.append(params_fit[name][0])

    cross_name = f"{names[0]}:{names[1]}"
    variograms[cross_name] = empirical_cross_cov(
        data_dict,
        space_lags,
        tol,
        time_lag,
        crop_lags=crop_lags,
        normalize=normalize_cov,
    )
    params_fit[cross_name], df_fit = fit_variogram_wls(
        variograms[cross_name]["lag"],
        variograms[cross_name][cross_name],
        variograms[cross_name]["counts"],
        cross_guess,
        sigmas=sigmas,
        covariogram=True,
        cross=True,
        normalized=normalize_cov,
    )
    variograms[cross_name] = pd.merge(
        variograms[cross_name], df_fit, on="lag", how="outer"
    )

    return variograms, params_fit

