import warnings

from numba import njit, prange
import numpy as np
import pandas as pd
import scipy.special as sps
from scipy.optimize import curve_fit, minimize

from krige_tools import distance_matrix

# TODO: establish a variogram class
SIG_L = 0.2
SIG_U = 0.9
NU_L = 0.2
NU_U = 3.5
LEN_L = 500
LEN_U = 1e3
NUG_L = 0.35
NUG_U = 0.8
RHO_L = -1.0
RHO_U = -0.7


def construct_variogram_bins(min_dist, max_dist, n_bins):
    bin_centers = np.linspace(min_dist, max_dist, n_bins)
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = np.arange(min_dist - 0.5 * bin_width, max_dist + bin_width, bin_width)
    # check that bin centers are actually centered
    if not np.allclose((bin_edges[1:] + bin_edges[:-1]) / 2, bin_centers):
        warnings.warn("WARNING: variogram bins are not centered.")
    bin_edges[0] = 0
    return bin_centers, bin_edges


def shift_longitude(coords):
    """Given an array of [[lat, lon]] in real degrees, add 0.5-degrees to longitude values."""
    coords_s = np.copy(coords)
    coords_s[:, 1] = coords_s[:, 1] + 0.5
    return coords_s


def distance_matrix_time(T1, T2, units="M"):
    """Computes the relative difference in months among all pairs of points given two sets of dates."""
    T1 = T1.astype(f"datetime64[{units}]")
    T2 = T2.astype(f"datetime64[{units}]")
    return np.abs(np.subtract.outer(T1, T2)).astype(float)


@njit
def spatial_pairs(D, bin_edges):
    """Returns indices of pairs within a given bin."""
    pairs = np.argwhere((D >= bin_edges[0]) & (D < bin_edges[1]))
    return pairs


@njit
def temporal_pairs(D, dist):
    """Returns backward temporal pairs from temporal distance matrix D."""
    # With a temporal offset of 'dist' already applied to the data, corrdinates will be along the diagonal.
    pairs = np.argwhere(D == dist)
    return pairs[pairs[:, 0] == pairs[:, 1]]


@njit
def spacetime_var_calc(pairs_time, pairs_space, d1, d2):
    """
    Computes the squared difference for each pair of spatial and temporal indices, and returns the mean of non-missing elements. Needs to be computed for each variogram bin.
    
    Parameters:
        pairs_time: Nx2 array with columns {time_id, time_id}
        pairs_space: Mx2 array with columns {location_id, location_id} 
        d1, d2: Kx3 arrays with columns {time_id, location_id, values}
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
            point_var1 = (d1[:, 0] == pairs_time[i, 0]) & (
                d1[:, 1] == pairs_space[j, 0]
            )
            point_var2 = (d2[:, 0] == pairs_time[i, 1]) & (
                d2[:, 1] == pairs_space[j, 1]
            )
            diff = d1[point_var1] - d2[point_var2]
            if diff.size:  # array is not empty, data available at matched points
                pairs_var[i, j] = (diff[0, 2]) ** 2
            else:
                pairs_var[i, j] = np.nan

    return 0.5 * np.nanmean(pairs_var), np.count_nonzero(~np.isnan(pairs_var))
    # return np.nanmean(pairs_var), np.count_nonzero(~np.isnan(pairs_var))


@njit
def spacetime_cov_calc(pairs_time, pairs_space, d1, d2):
    """
    Computes the product for each pair of spatial and temporal indices, and returns the mean of non-missing elements. Needs to be computed for each variogram bin.
    
    Parameters:
        pairs_time: Nx2 array with columns {time_id for x1, time_id for x2}
        pairs_space: Mx2 array with columns {location_id for x1, location_id for x2}
        d1, d2: Kx3 arrays with columns {time_id, location_id, values}
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
def apply_bin_calcs(bin_edges, dist_space, pairs_time, data1, data2, covariogram):
    """For a fixed temporal lag, run (co)variogram calculations for each spatial bin in parallel."""
    variogram = np.zeros(bin_edges.size - 1)
    counts = np.zeros(bin_edges.size - 1)

    for i in prange(len(variogram)):  # pylint: disable=not-an-iterable
        pairs_space = spatial_pairs(dist_space, [bin_edges[i], bin_edges[i + 1]])
        if pairs_space.shape[0] == 0:
            print("Degenerate.")
        if covariogram:
            variogram[i], counts[i] = spacetime_cov_calc(
                pairs_time, pairs_space, data1, data2
            )
        else:
            variogram[i], counts[i] = spacetime_var_calc(
                pairs_time, pairs_space, data1, data2
            )

    return variogram, counts


def empirical_variogram(
    df,
    name,
    time_lag,
    n_bins=15,
    covariogram=False,
    shift_coords=False,
    fast_dist=False,
):
    """Basic function to compute a variogram from a dataframe."""
    # TODO: we can remove the time_lag arg right?
    # Establish space-time domain
    times = np.unique(df["time"].values)
    coords = np.unique(df[["lat", "lon"]].values, axis=0)

    # Precompute distances
    dist_time = distance_matrix_time(times, times)
    if shift_coords:
        dist_space = distance_matrix(
            coords, shift_longitude(coords), fast_dist=fast_dist
        )
    else:
        dist_space = distance_matrix(coords, coords, fast_dist=fast_dist)

    assert time_lag <= dist_time.max()
    bin_centers, bin_edges = construct_variogram_bins(
        dist_space.min(), 0.6 * dist_space.max(), n_bins
    )

    # Get temporal pairs
    pairs_time = temporal_pairs(dist_time, time_lag)

    # Format data and variogram dataframe
    data = df[["t_id", "loc_id"] + [name]].values
    df_vario = pd.DataFrame({"lag": bin_centers})

    # Compute variogram
    df_vario[name], df_vario["counts"] = apply_bin_calcs(
        bin_edges, dist_space, pairs_time, data, data, covariogram
    )
    if (df_vario["counts"] < 30).any():
        warnings.warn(
            f"WARNING: Fewer than 30 pairs used for at least one bin in variogram calculation."
        )

    return df_vario


def empirical_cross_variogram(
    data_dict,
    time_lag,
    n_bins=15,
    covariogram=False,
    shift_coords=False,
    fast_dist=False,
):
    """Basic function to compute a (co)variogram from a pair of dataframes stored in dict. If dataframes are not identical, this will be a cross (co)variogram."""
    names = list(data_dict.keys())
    assert len(names) == 2

    # Establish space-time domains
    times = [np.unique(data_dict[name]["time"].values) for name in names]
    coords = [
        np.unique(data_dict[name][["lat", "lon"]].values, axis=0) for name in names
    ]
    if shift_coords:
        coords[1] = shift_longitude(coords[1])

    # Precompute distances
    dist_time = distance_matrix_time(times[0], times[1])
    dist_space = distance_matrix(coords[0], coords[1], fast_dist=fast_dist)

    assert time_lag <= dist_time.max()
    bin_centers, bin_edges = construct_variogram_bins(
        dist_space.min(), 0.6 * dist_space.max(), n_bins
    )

    # Get directional temporal pairs (don't assume symmetry)
    pairs_time = temporal_pairs(dist_time, time_lag)

    # Format data and variogram dataframe
    data = [data_dict[name][["t_id", "loc_id"] + [name]].values for name in names]
    df_cross = pd.DataFrame({"lag": bin_centers})

    # Compute cross-(co)variogram
    cross_name = f"{names[0]}:{names[1]}"
    df_cross[cross_name], df_cross["counts"] = apply_bin_calcs(
        bin_edges, dist_space, pairs_time, data[0], data[1], covariogram
    )
    # if normalize:
    #     assert sigmas is not None
    #     df_cross[cross_name] = df_cross[cross_name] / np.prod(sigmas)
    if (df_cross["counts"] < 30).any():
        warnings.warn(
            f"WARNING: Fewer than 30 pairs used for at least one bin in covariogram calculation."
        )

    return df_cross


# TODO: add ability to `freeze` parameters
def matern_correlation(xdata, nu, len_scale):
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


# NOTE: this parameterization requires longer length scales when fitting (which seems counter intuitive)
# def matern_correlation(xdata, nu, len_scale):
#     xdata_ = xdata[xdata > 0.0] / len_scale
#     corr = np.ones_like(xdata)
#     corr[xdata > 0.0] = np.exp(
#         (1.0 - nu) * np.log(2) - sps.gammaln(nu) + nu * np.log(xdata_)
#     ) * sps.kv(nu, xdata_)

#     corr[np.logical_not(np.isfinite(corr))] = 0.0
#     # normalized Matern is positive
#     corr = np.maximum(corr, 0.0)
#     return corr


def matern_vario(xdata, sigma, nu, len_scale, nugget):
    return sigma ** 2 * (1.0 - matern_correlation(xdata, nu, len_scale)) + nugget


def matern_cross_vario(xdata, sigmas, nuggets, nu, len_scale, rho):
    sill = 0.5 * (sigmas[0] ** 2 + sigmas[1] ** 2 + nuggets[0] + nuggets[1])
    return sill - rho * sigmas[0] * sigmas[1] * matern_correlation(xdata, nu, len_scale)


def matern_cov(xdata, sigma, nu, len_scale, nugget):
    cov = sigma ** 2 * matern_correlation(xdata, nu, len_scale)
    cov[xdata == 0] += nugget
    return cov


def matern_cross_cov(xdata, sigmas, nu, len_scale, rho):
    return rho * sigmas[0] * sigmas[1] * matern_correlation(xdata, nu, len_scale)


def weighted_least_squares(ydata, yfit, bin_counts):
    zeros = np.argwhere(yfit == 0.0)
    non_zero = np.argwhere(yfit != 0.0)
    wls = np.zeros_like(yfit)
    wls[zeros] = (
        bin_counts[zeros] * ydata[zeros] ** 2
    )  # NOTE: is this wrong to do? (for variograms with nonzero nugget, it won't even come up)
    wls[non_zero] = (
        bin_counts[non_zero]
        * ((ydata[non_zero] - yfit[non_zero]) / yfit[non_zero]) ** 2
    )
    return np.sum(wls)


def wls_cost(params, xdata, ydata, bin_counts, sigmas, nuggets):
    if sigmas is not None:
        yfit = matern_cross_vario(xdata, sigmas, nuggets, *params)
    else:
        yfit = matern_vario(xdata, *params)
    return weighted_least_squares(ydata, yfit, bin_counts)


# def fit_variogram_wls(
#     xdata,
#     ydata,
#     bin_counts,
#     initial_guess,
#     sigmas=None,
#     covariogram=True,
#     cross=False,
#     normalized=False,
# ):
#     """
#     Fit covariance parameters to empirical (co)variogram by weighted least squares (Cressie, 1985).

#     Parameters:
#         xdata: pd.series giving the spatial lags
#         ydata: pd.series giving the empirical variogram values to be fitted
#         bin_counts: pd.series indicating the number of spatio-temporal point pairs used to calculate each empirical value
#         initial_guess: list of parameter starting values given as one of [sigma, nu, len_scale, nugget] or [nu, len_scale, rho]
#         sigmas: list of standard deviations if fitting a cross covariance
#         cross: cross dependence?
#         normalized: correlation (True), covariance (False)
#     Returns:
#         params: list of parameter values
#         fit: the theoretical (co)variogram fit
#     """
#     pred = np.linspace(0, 1.1 * xdata.max(), 100)
#     if covariogram:
#         if cross:
#             assert len(initial_guess) == 3
#             bounds = [(NU_L, NU_U), (LEN_L, LEN_U), (RHO_L, RHO_U)]
#             if normalized:
#                 # Cross correlation, fit cross correlogram
#                 optim_result = minimize(
#                     wls_cost_norm,
#                     initial_guess,
#                     args=(xdata.values, ydata.values, bin_counts.values, True),
#                     method="L-BFGS-B",
#                     bounds=bounds,
#                 )
#                 fit = matern_cross_corr(pred, *optim_result.x)
#             else:
#                 # Cross covariance, fit cross covariogram
#                 assert sigmas is not None
#                 optim_result = minimize(
#                     wls_cost,
#                     initial_guess,
#                     args=(
#                         xdata.values,
#                         ydata.values,
#                         bin_counts.values,
#                         sigmas,
#                         covariogram,
#                     ),
#                     method="L-BFGS-B",
#                     bounds=bounds,
#                 )
#                 fit = matern_cross_cov(pred, sigmas, *optim_result.x)
#         else:
#             if normalized:
#                 # Univariate correlation, fit correlogram
#                 assert len(initial_guess) == 2
#                 bounds = [(NU_L, NU_U), (LEN_L, LEN_U)]
#                 optim_result = minimize(
#                     wls_cost_norm,
#                     initial_guess,
#                     args=(xdata.values, ydata.values, bin_counts.values, False),
#                     method="L-BFGS-B",
#                     bounds=bounds,
#                 )
#                 fit = matern_correlation(pred, *optim_result.x)
#             else:
#                 # Univariate covariance, fit covariogram
#                 assert len(initial_guess) == 4
#                 bounds = [(SIG_L, SIG_U), (NU_L, NU_U), (LEN_L, LEN_U), (NUG_L, NUG_U)]
#                 optim_result = minimize(
#                     wls_cost,
#                     initial_guess,
#                     args=(
#                         xdata.values,
#                         ydata.values,
#                         bin_counts.values,
#                         None,
#                         covariogram,
#                     ),
#                     method="L-BFGS-B",
#                     bounds=bounds,
#                 )
#                 # fit = matern_vario(xdata, *optim_result.x)
#                 fit = matern_cov(pred, *optim_result.x)
#     else:
#         # Univariate covariance, fit variogram
#         assert len(initial_guess) == 3
#         bounds = [(SIG_L, SIG_U), (NU_L, NU_U), (LEN_L, LEN_U), (NUG_L, NUG_U)]
#         optim_result = minimize(
#             wls_cost,
#             initial_guess,
#             args=(xdata.values, ydata.values, bin_counts.values, None, covariogram),
#             method="L-BFGS-B",
#             bounds=bounds,
#         )
#         fit = matern_vario(pred, *optim_result.x)

#     if optim_result.success == False:
#         print("ERROR: optimization did not converge.")
#         warnings.warn("ERROR: optimization did not converge.")

#     return optim_result.x, pd.DataFrame({"lag": pred, "wls_fit": fit})


def fit_variogram_wls(
    xdata, ydata, bin_counts, initial_guess, sigmas=None, nuggets=None
):
    """
    Fit covariance parameters to empirical variogram by weighted least squares (Cressie, 1985).
    
    Parameters:
        xdata: pd.series giving the spatial lags
        ydata: pd.series giving the empirical variogram values to be fitted
        bin_counts: pd.series indicating the number of spatio-temporal point pairs used to calculate each empirical value
        initial_guess: list of parameter starting values given as one of [sigma, nu, len_scale, nugget] or [nu, len_scale, rho]
        sigmas: list of standard deviations if fitting a cross (co)variogram
    Returns:
        params: list of parameter values
        fit: dataframe containing the theoretical covariance
    """
    pred = np.linspace(0, 1.1 * xdata.max(), 100)
    if sigmas is not None:
        # Cross covariance, fit cross-variogram
        assert len(initial_guess) == 3
        bounds = [(NU_L, NU_U), (LEN_L, LEN_U), (RHO_L, RHO_U)]
        optim_result = minimize(
            wls_cost,
            initial_guess,
            args=(xdata.values, ydata.values, bin_counts.values, sigmas, nuggets),
            method="L-BFGS-B",
            bounds=bounds,
        )
        var_fit = matern_cross_vario(pred, sigmas, nuggets, *optim_result.x)
        cov_fit = matern_cross_cov(pred, sigmas, *optim_result.x)
    else:
        # Univariate covariance, fit variogram
        assert len(initial_guess) == 4
        bounds = [(SIG_L, SIG_U), (NU_L, NU_U), (LEN_L, LEN_U), (NUG_L, NUG_U)]
        optim_result = minimize(
            wls_cost,
            initial_guess,
            args=(xdata.values, ydata.values, bin_counts.values, None, None),
            method="L-BFGS-B",
            bounds=bounds,
        )
        var_fit = matern_vario(pred, *optim_result.x)
        cov_fit = matern_cov(pred, *optim_result.x)

    if optim_result.success == False:
        print("ERROR: optimization did not converge.")
        warnings.warn("ERROR: optimization did not converge.")

    return (
        optim_result.x,
        pd.DataFrame({"lag": pred, "wls_fit": var_fit}),
        pd.DataFrame({"lag": pred, "wls_fit": cov_fit}),
    )


def check_cauchyshwarz(covariograms, names):
    """Check the Cauchy-Shwarz inequality."""
    name1 = names[0]
    name2 = names[1]
    cross_name = f"{name1}:{name2}"

    # NOTE: Not exactly C-S if minimum lag is not 0, but should be close
    cov1_0 = covariograms[name1][
        covariograms[name1]["lag"] == np.min(covariograms[name1]["lag"])
    ][name1][0]
    cov2_0 = covariograms[name2][
        covariograms[name2]["lag"] == np.min(covariograms[name2]["lag"])
    ][name2][0]
    cross_cov = covariograms[cross_name][cross_name] ** 2

    if np.any(cross_cov > cov1_0 * cov2_0):
        warnings.warn("WARNING: Cauchy-Shwarz inequality not upheld.")


def variogram_analysis(
    mf, cov_guesses, cross_guess, n_bins=15, shift_coords=False,
):
    """
    Compute the empirical spatio-temporal variograms from a multi-field object and find the weighted least squares fit.
    NOTE: data must have the same scale and time-indexed spatial mean is assumed to be zero

    Parameters:
        mf: multi-field object
        cov_guesses: covariance params initial guess for WLS fit; list [[sigma, nu, len_scale, nugget], [sigma, nu, len_scale, nugget]]
        cross_guess: cross-cov parmas initial guess for WLS fit; list [nu, len_scale, rho]
        tol [deprecated]: radius of the spatial neighborhood into which data point pairs are grouped for semivariance estimates; note that this can be seen as a rolling window so depending on the size, some pairs may be repeated in multiple bins 
        crop_lags [deprecated]: should spatial lag vector be trimmed to a fraction of the maximum distance, and formatted such that the first non-zero element is at least the minimum distance?
        n_bins: number of bins into which point pairs are grouped for variogram estimates

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

    # for name in names:
    #     if standardize:
    #         # Standardize locally (temporal replication)
    #         data_dict[name][name] = (
    #             data_dict[name]
    #             .groupby("loc_id")[name]
    #             .transform(lambda x: (x - x.mean()) / x.std())
    #         )
    #     # Remove the time-indexed spatial mean
    #     data_dict[name][name] = (
    #         data_dict[name].groupby("t_id")[name].transform(lambda x: x - x.mean())
    #     )

    # Compute and fit variograms and covariograms
    variograms = dict()
    covariograms = dict()
    params_fit = dict()
    sigmas = list()
    nuggets = list()
    for i, name in enumerate(names):
        print(name)
        # NOTE: no temporal lag in variograms/covariograms
        variograms[name] = empirical_variogram(
            data_dict[name],
            name,
            0,
            n_bins=n_bins,
            covariogram=False,
            shift_coords=shift_coords,
            fast_dist=mf.fast_dist,
        )
        covariograms[name] = empirical_variogram(
            data_dict[name],
            name,
            0,
            n_bins=n_bins,
            covariogram=True,
            shift_coords=shift_coords,
            fast_dist=mf.fast_dist,
        )
        params_fit[name], var_fit, cov_fit = fit_variogram_wls(
            variograms[name]["lag"],
            variograms[name][name],
            variograms[name]["counts"],
            cov_guesses[i],
        )
        sigmas.append(params_fit[name][0])
        nuggets.append(params_fit[name][-1])
        variograms[name] = pd.merge(variograms[name], var_fit, on="lag", how="outer")
        covariograms[name] = pd.merge(
            covariograms[name], cov_fit, on="lag", how="outer"
        )

    # Compute and fit cross-variogram and cross-covariogram
    cross_name = f"{names[0]}:{names[1]}"
    print(cross_name)
    variograms[cross_name] = empirical_cross_variogram(
        data_dict,
        time_lag,
        n_bins=n_bins,
        covariogram=False,
        shift_coords=shift_coords,
        fast_dist=mf.fast_dist,
    )
    covariograms[cross_name] = empirical_cross_variogram(
        data_dict,
        time_lag,
        n_bins=n_bins,
        covariogram=True,
        shift_coords=shift_coords,
        fast_dist=mf.fast_dist,
    )
    params_fit[cross_name], var_fit, cov_fit = fit_variogram_wls(
        variograms[cross_name]["lag"],
        variograms[cross_name][cross_name],
        variograms[cross_name]["counts"],
        cross_guess,
        sigmas=sigmas,
        nuggets=nuggets,
    )
    variograms[cross_name] = pd.merge(
        variograms[cross_name], var_fit, on="lag", how="outer"
    )
    covariograms[cross_name] = pd.merge(
        covariograms[cross_name], cov_fit, on="lag", how="outer"
    )
    # TODO: sort out how to handle different data and prediction domains
    # check_cauchyshwarz(variograms, names)

    return variograms, covariograms, params_fit

