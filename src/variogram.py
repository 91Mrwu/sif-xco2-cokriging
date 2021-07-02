import warnings

# from numba import njit, prange
import numpy as np
import pandas as pd
import scipy.special as sps
from scipy.optimize import minimize

from spatial_tools import distance_matrix

# TODO: establish a variogram class
SIG_L = 0.8
SIG_U = 1.4
NU_L = 0.2
NU_U = 3.5
LEN_L = 1e2
LEN_U = 1e3
NUG_L = 0.0
NUG_U = 0.1
RHO_L = -1.0
RHO_U = -0.1


def construct_variogram_bins(min_dist, max_dist, n_bins):
    bin_centers = np.linspace(min_dist, max_dist, n_bins)
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = np.arange(min_dist - 0.5 * bin_width, max_dist + bin_width, bin_width)
    # check that bin centers are actually centered
    if not np.allclose((bin_edges[1:] + bin_edges[:-1]) / 2, bin_centers):
        warnings.warn("WARNING: variogram bins are not centered.")
    bin_edges[0] = 0
    return bin_centers, bin_edges


# @njit
def cloud_calc(values1, values2, covariogram):
    """Calculate the semivariogram or covariogram for all point pairs."""
    resid1 = values1 - values1.mean()
    resid2 = values2 - values2.mean()
    if covariogram:
        cloud = np.multiply.outer(resid1, resid2)
    else:
        cloud = 0.5 * (np.subtract.outer(resid1, resid2)) ** 2
    return cloud


def variogram_cloud(dist, values1, values2=None, covariogram=False):
    """Calculate the (cross-) semivariogram cloud from a distance matrix and corresponding points."""
    if values2 is not None:
        dist = dist.flatten()
        cloud = cloud_calc(values1, values2, covariogram).flatten()
    else:
        idx = np.triu_indices(dist.shape[0], k=1, m=dist.shape[1])
        dist = dist[idx]
        cloud = cloud_calc(values1, values1, covariogram)[idx]

    assert cloud.shape == dist.shape
    return pd.DataFrame({"distance": dist, "variogram": cloud})


# def microlag_clouds(df_group, data_name, fast_dist=True):
#     """For a given lat-lon group in a microlag dataframe, compute all semivariogram and cross-semivariogram clouds. Return in columns of shared dataframe."""
#     coords = df_group[["lat", "lon"]].values
#     dist = distance_matrix(coords, coords, fast_dist=fast_dist)
#     values = df_group[data_name].values
#     return variogram_cloud(dist, values)

#     # # How to incorporate cross-semivariogram into microlag cloud?
#     # cloud_sif = variogram_cloud(dist, values_sif)
#     # cloud_cross = variogram_cloud(dist, values_xco2, values2=values_sif)

#     # return pd.DataFrame(
#     #     {
#     #         "distance": cloud_xco2.distance,
#     #         "variogram_xco2": cloud_xco2.variogram,
#     #         "variogram_sif": cloud_sif.variogram,
#     #         "variogram_cross": cloud_cross.variogram,
#     #     }
#     # )


def empirical_variogram(
    dist, values1, values2=None, n_bins=50, covariogram=False, label=None
):
    """Compute the empirical semivariogram or covariogram and return as a dataframe with bin averages and counts. If values2 is not None, this will be a cross-variogram."""
    df = variogram_cloud(dist, values1, values2=values2, covariogram=covariogram)
    # NOTE: if computation becomes slow, this could be done before computing the cloud values
    df = df[df.distance <= 0.5 * dist.max()]

    bin_centers, bin_edges = construct_variogram_bins(
        df["distance"].min(), df["distance"].max(), n_bins
    )
    df["bin_center"] = pd.cut(
        df["distance"], bin_edges, labels=bin_centers, include_lowest=True
    )
    df = (
        df.groupby("bin_center")["variogram"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "bin_mean"})
        .reset_index()
    )
    # convert bins from categories to numeric
    df["bin_center"] = df["bin_center"].astype("string").astype("float")
    if (df["count"] < 30).any():
        warnings.warn(
            f"WARNING: Fewer than 30 pairs used for at least one bin in variogram calculation."
        )
    if label is not None:
        df["label"] = label
    return df


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


def composite_wls(params: np.array, df_comp: pd.DataFrame) -> np.float:
    """Composite WLS cost function."""
    sigmas = params[[0, -4]]
    nuggets = params[[3, -1]]
    df_comp["fit"] = np.nan
    df_comp.loc[df_comp["label"] == "y1", "fit"] = matern_vario(
        df_comp.loc[df_comp["label"] == "y1", "bin_center"].values, *params[:4]
    )
    df_comp.loc[df_comp["label"] == "y2", "fit"] = matern_vario(
        df_comp.loc[df_comp["label"] == "y2", "bin_center"].values, *params[-4:]
    )
    df_comp.loc[df_comp["label"] == "cross", "fit"] = matern_cross_vario(
        df_comp.loc[df_comp["label"] == "cross", "bin_center"].values,
        sigmas,
        nuggets,
        *params[4:7],
    )
    return weighted_least_squares(
        df_comp["bin_mean"].values, df_comp["fit"].values, df_comp["count"].values
    )


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
        warnings.warn("ERROR: optimization did not converge.")

    return (
        optim_result.x,
        pd.DataFrame({"lag": pred, "wls_fit": var_fit}),
        pd.DataFrame({"lag": pred, "wls_fit": cov_fit}),
    )


def composite_fit(params: np.array, df_comp: pd.DataFrame) -> np.array:
    """Composite WLS fits marginal and cross-semivariogram parameters simultaneously.
    
    Parameters:
        params: initial guess [sigma_1, nu_1, len_scale_1, tau_1, nu_12, len_scale_12, rho_12, sigma_2, nu_2, len_scale_2, tau_2]
        df_comp: labelled empirical (cross-) semivariograms stacked into a single dataframe
    Returns:
        optimal parameters
    """
    assert len(params) == 11
    bounds_marg = [(SIG_L, SIG_U), (NU_L, NU_U), (LEN_L, LEN_U), (NUG_L, NUG_U)]
    bounds_cross = [(NU_L, NU_U), (LEN_L, LEN_U), (RHO_L, RHO_U)]
    bounds = bounds_marg + bounds_cross + bounds_marg
    optim_result = minimize(
        composite_wls, params, args=(df_comp), method="L-BFGS-B", bounds=bounds,
    )
    if optim_result.success == False:
        warnings.warn("ERROR: optimization did not converge.")
    return optim_result.x


def composite_predict(params: np.array, df_comp: pd.DataFrame) -> tuple:
    """Produces semivariogram and covariogram models using fitted parameters."""
    sigmas = params[[0, -4]]
    nuggets = params[[3, -1]]

    pred = np.linspace(0, df_comp["bin_center"].max(), 100)
    df_var = pd.DataFrame({"distance": pred})
    df_cov = df_var.copy()

    df_var["fit_1"] = matern_vario(pred, *params[:4])
    df_cov["fit_1"] = matern_cov(pred, *params[:4])
    df_var["fit_2"] = matern_vario(pred, *params[-4:])
    df_cov["fit_2"] = matern_cov(pred, *params[-4:])
    df_var["fit_cross"] = matern_cross_vario(pred, sigmas, nuggets, *params[4:7])
    df_cov["fit_cross"] = matern_cross_cov(pred, sigmas, *params[4:7])

    return df_var, df_cov


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
    mf, params_guess, n_bins=50,
):
    """
    Compute the empirical spatial-only variograms from a multi-field object and find the weighted least squares fit.
    NOTE: 
    - data must have the same scale and the spatial mean is assumed to be zero

    Parameters:
        mf: multi-field object
        params_guess: covariance params initial guess [sigma_1, nu_1, len_scale_1, tau_1, nu_12, len_scale_12, rho_12, sigma_2, nu_2, len_scale_2, tau_2]
        n_bins: number of bins into which point pairs are grouped for variogram estimates

    Returns:
        variograms: dictionary containing semivariogram and cross-semivariogram dataframes
        covariograms: dictionary containing covariogram and cross-covariogram dataframes 
        params_fit: dictionary of parameter fits for each semivariogram and cross-semivariogram
    """
    fields = [mf.field_1, mf.field_2]
    dists = [
        distance_matrix(mf.field_1.coords, mf.field_1.coords, fast_dist=mf.fast_dist,),
        distance_matrix(mf.field_2.coords, mf.field_2.coords, fast_dist=mf.fast_dist,),
    ]
    dist_cross = distance_matrix(
        mf.field_1.coords, mf.field_2.coords, fast_dist=mf.fast_dist
    )

    # Compute and semivariograms and covariograms
    variograms = dict()
    covariograms = dict()
    labels = ["y1", "y2"]
    for i, field in enumerate(fields):
        variograms[field.data_name] = empirical_variogram(
            dists[i], field.values, n_bins=n_bins, covariogram=False, label=labels[i]
        )
        covariograms[field.data_name] = empirical_variogram(
            dists[i], field.values, n_bins=n_bins, covariogram=True,
        )

    # Compute cross-semivariogram and cross-covariogram
    cross_name = f"{fields[0].data_name}:{fields[1].data_name}"
    variograms[cross_name] = empirical_variogram(
        dist_cross,
        fields[0].values,
        values2=fields[1].values,
        n_bins=n_bins,
        covariogram=False,
        label="cross",
    )
    covariograms[cross_name] = empirical_variogram(
        dist_cross,
        fields[0].values,
        values2=fields[1].values,
        n_bins=n_bins,
        covariogram=True,
    )

    # Fit model parameters and produce predicted values
    if params_guess is not None:
        df_comp = pd.concat(variograms.values())
        params_fit = composite_fit(params_guess, df_comp)
        variograms["fit"], covariograms["fit"] = composite_predict(params_fit, df_comp)
    else:
        params_fit = None

    # TODO: sort out how to handle different data and prediction domains
    # check_cauchyshwarz(variograms, names)

    return variograms, covariograms, params_fit

