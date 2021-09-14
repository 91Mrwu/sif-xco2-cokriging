import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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


def variogram_analysis(mf, params_guess, n_bins=50, max_dist=None):
    """Compute the empirical spatial-only variograms from a multi-field object and find the weighted least squares fit.

    Parameters:
        mf: multi-field object
        params_guess: covariance params initial guess [sigma_1, nu_1, len_scale_1, tau_1, nu_12, len_scale_12, rho_12, sigma_2, nu_2, len_scale_2, tau_2]
        n_bins: number of bins into which point pairs are grouped for variogram estimates

    Returns:
        variograms: dictionary containing semivariogram and cross-semivariogram dataframes
        covariograms: dictionary containing covariogram and cross-covariogram dataframes
        params_fit: dictionary of parameter fits for each semivariogram and cross-semivariogram

    NOTE:
    - data must have the same scale and the spatial mean is assumed to be zero
    - observation and prediction domains are assumed to be the same (even between datasets)
    """
    fields = [mf.field_1, mf.field_2]
    dists = [
        distance_matrix(
            mf.field_1.coords,
            mf.field_1.coords,
            fast_dist=mf.fast_dist,
        ),
        distance_matrix(
            mf.field_2.coords,
            mf.field_2.coords,
            fast_dist=mf.fast_dist,
        ),
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
            dists[i],
            field.values,
            n_bins=n_bins,
            max_dist=max_dist,
            covariogram=False,
            label=labels[i],
        )
        covariograms[field.data_name] = empirical_variogram(
            dists[i],
            field.values,
            n_bins=n_bins,
            max_dist=max_dist,
            covariogram=True,
        )

    # Compute cross-semivariogram and cross-covariogram
    cross_name = f"{fields[0].data_name}:{fields[1].data_name}"
    variograms[cross_name] = empirical_variogram(
        dist_cross,
        fields[0].values,
        values2=fields[1].values,
        n_bins=n_bins,
        max_dist=max_dist,
        covariogram=False,
        label="cross",
    )
    covariograms[cross_name] = empirical_variogram(
        dist_cross,
        fields[0].values,
        values2=fields[1].values,
        n_bins=n_bins,
        max_dist=max_dist,
        covariogram=True,
    )

    # Fit model parameters and produce predicted values
    if params_guess is not None:
        df_comp = pd.concat(variograms.values())
        params_fit = composite_fit(params_guess, df_comp)
        variograms["fit"], covariograms["fit"] = composite_predict(params_fit, df_comp)
    else:
        params_fit = None

    # TODO: check paramter validity
    # check_cauchyshwarz(variograms, names)

    return variograms, covariograms, params_fit
