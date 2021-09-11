"""
This will be the main module.
It will define the univariate and bivariate models, along with parameter bounds (getter and setter)
It will provide an interface to model fitting using a variography class (with args)
It will provide an interface to model prediction using a cokriging class (with args, mainly prediction locations)
"""
import numpy as np
import pandas as pd

import scipy.special as sps

# from scipy.linalg import cho_factor, cho_solve, LinAlgError
# from scipy.optimize import minimize

import spatial_tools
import variography as vg


class MarginalParam:
    """Multivariate marginal Matern covariance parameter."""

    def __init__(self, name: str, default: float, bounds: tuple, dim: int = 2) -> None:
        self.name = name
        self.dim = dim
        self.type = type
        self.default = default
        self.bounds = bounds
        self.values = np.nan * np.zeros((dim, dim))
        np.fill_diagonal(self.values, default)

    def get_names(self):
        return [f"{self.name}_{i+1}{i+1}" for i in range(self.dim)]

    def get_values(self):
        return self.values.diagonal()

    def set_values(self, x: np.ndarray):
        np.fill_diagonal(self.values, x)
        return self

    def to_dataframe(self):
        df = (
            pd.DataFrame.from_dict(
                dict(zip(self.get_names(), self.get_values())),
                orient="index",
                columns=["value"],
            )
            .reset_index()
            .rename(columns={"index": "name"})
        )
        df["bounds"] = [self.bounds] * len(df)
        return df


class CrossParam(MarginalParam):
    """Multivariate Matern covariance parameter with cross dependence."""

    def __init__(self, name: str, default: float, bounds: tuple, dim: int = 2) -> None:
        super().__init__(name, default, bounds, dim=dim)
        self._triu_index = np.triu_indices(dim)
        self.values[self._triu_index] = default

    def get_names(self):
        return [
            f"{self.name}_{i+1}{j+1}"
            for i in range(self.dim)
            for j in range(self.dim)
            if i <= j
        ]

    def get_values(self):
        return self.values[self._triu_index]

    def set_values(self, x: np.ndarray):
        self.values[self._triu_index] = x
        return self


class RhoParam(MarginalParam):
    """Multivariate Matern covariance parameter with cross dependence only."""

    def __init__(self, name: str, default: float, bounds: tuple, dim: int = 2) -> None:
        super().__init__(name, default, bounds, dim=dim)
        self._triu_index = np.triu_indices(dim, k=1)
        self.values[self._triu_index] = default

    def get_names(self):
        return [
            f"{self.name}_{i+1}{j+1}"
            for i in range(self.dim)
            for j in range(self.dim)
            if i < j
        ]

    def get_values(self):
        return self.values[self._triu_index]

    def set_values(self, x: np.ndarray):
        self.values[self._triu_index] = x
        return self


class MaternParams:
    """Multivariate Matern covariance parameters.

    Formulation:
        sigma: process specific standard deviation
        nu: process specific smoothess
        len_scale: process specific length scale
        nugget: process specific squared nugget (i.e., tau^2)
        rho: co-located cross-correlation coefficient
    """

    def __init__(self, n_procs: int = 2) -> None:
        self.n_procs = n_procs

        # Collect all the params here


class MaternCovariance:
    """Matern covariance model (Rassmussen and Williams, 2006)."""

    def __init__(self, params: list[float] = None):
        pass

    def correlation(self, h: np.ndarray):
        return matern_correlation(h, *self.params[[1, 2]])

    def covariance(self, h: np.ndarray):
        """Matern covariance function."""
        cov = self.params[0] ** 2 * self.correlation(h)
        cov[h == 0] += self.params[3]
        return cov


class CrossCovariance:
    """Matern cross-covariance structure."""

    def __init__(self, K1, K2):
        self.param_names = ["nu", "len_scale", "rho"]
        self.param_bounds = [(0.2, 3.5), (1e2, 2e3), (-1.0, 1.0)]
        nu = np.mean([K1.params[1], K2.params[1]])
        len_scale = np.mean([K1.params[2], K2.params[2]])

    def covariance(self, h: np.ndarray):
        pass


# TODO: add method to check parameter validity (overall pos def and paper specs)
class FullBivariateMatern:
    """Full bivariate Matern covariance model (Gneiting et al., 2010).

    Parameterization (see MaternCovariance)
    """

    def __init__(self, marginal_kernals: list[MaternCovariance] = None):
        p = len(marginal_kernals)
        if marginal_kernals is not None:
            self.marginal_kernals = marginal_kernals
        else:
            self.marginal_kernals = [MaternCovariance(), MaternCovariance()]

        # start data frame of parameters (this will be the main api access to parameters)

        # initalize cross-covariance and append params to table
        # NOTE: use mean down columns
        # self.cross_kernel = _avg_cross_params(self.marginal_kernals)

        # get list of relabeled param names
        # get updated list of param bounds
        # get 1d array of params for optimization

        # set up a parameter updater

    def cross_covariance(self, h: np.ndarray):
        return (
            self.params.rho
            * self.params.sigmas[0]
            * self.params.sigmas[1]
            * self.correlation(h)
        )

    def pred_covariance(self, dist_mat):
        """Computes the variance-covariance matrix for prediction location(s).

        NOTE: if nugget is not added here, then cov model needs to be updated in notation
        """
        return self.kernel_1.sigma ** 2 * self.kernel_1.correlation(dist_mat)

    def cross_covariance(self, dist_blocks):
        """Computes the cross-covariance vectors for prediction distances."""
        c_11 = self.kernel_1.sigma ** 2 * self.kernel_1.correlation(
            dist_blocks["block_11"]
        )
        c_12 = (
            self.kernel_1.sigma
            * self.kernel_2.sigma
            * self.kernel_b.correlation(dist_blocks["block_12"])
        )
        cov_vecs = np.hstack((c_11, c_12))
        # normalize rows of cov_vecs with joint_std_inverse via broadcasting
        assert (
            cov_vecs.shape[1] == self.fields.joint_std_inverse.shape[0]
        ), "mismatched dimensions"
        return cov_vecs * self.fields.joint_std_inverse

    def covariance_matrix(self, dist_blocks):
        """Constructs the bivariate Matern covariance matrix.

        NOTE: ask about when to add nugget and error variance along diag
        """
        C_11 = self.kernel_1.sigma ** 2 * self.kernel_1.correlation(
            dist_blocks["block_11"]
        )
        C_12 = (
            self.rho
            * self.kernel_1.sigma
            * self.kernel_2.sigma
            * self.kernel_b.correlation(dist_blocks["block_12"])
        )
        C_21 = (
            self.rho
            * self.kernel_1.sigma
            * self.kernel_2.sigma
            * self.kernel_b.correlation(dist_blocks["block_21"])
        )
        C_22 = self.kernel_2.sigma ** 2 * self.kernel_2.correlation(
            dist_blocks["block_22"]
        )

        # add nugget and measurement error variance along diagonals
        np.fill_diagonal(C_11, C_11.diagonal() + self.kernel_1.nugget + self.sigep_11)
        np.fill_diagonal(C_22, C_22.diagonal() + self.kernel_2.nugget + self.sigep_22)

        # stack blocks into joint covariance matrix and normalize by standard deviation
        cov_mat = np.block([[C_11, C_12], [C_21, C_22]])
        return spatial_tools.pre_post_diag(self.fields.joint_std_inverse, cov_mat)

    def set_params(self, params_arr):
        """Set model parameters."""
        self.kernel_1.sigma = params_arr[0]
        self.kernel_1.nu = params_arr[1]
        self.kernel_1.len_scale = params_arr[2]
        self.kernel_1.nugget = params_arr[3]
        self.kernel_b.nu = params_arr[4]
        self.kernel_b.len_scale = params_arr[5]
        self.rho = params_arr[6]
        self.kernel_2.sigma = params_arr[7]
        self.kernel_2.nu = params_arr[8]
        self.kernel_2.len_scale = params_arr[9]
        self.kernel_2.nugget = params_arr[10]

    def set_param_bounds(self, bounds):
        """Set default parameter bounds using dictionary of lists."""
        self.param_bounds.update(bounds)

    def get_params(self):
        """Return model parameters as a dict."""
        return {
            "sigma_11": self.kernel_1.sigma,
            "nu_11": self.kernel_1.nu,
            "len_scale_11": self.kernel_1.len_scale,
            "nugget_11": self.kernel_1.nugget,
            # "sigep_11": self.sigep_11,
            "nu_12": self.kernel_b.nu,
            "len_scale_12": self.kernel_b.len_scale,
            "rho": self.rho,
            "sigma_22": self.kernel_2.sigma,
            "nu_22": self.kernel_2.nu,
            "len_scale_22": self.kernel_2.len_scale,
            "nugget_22": self.kernel_2.nugget,
            # "sigep_22": self.sigep_22,
        }

    def empirical_variograms(self, params_guess, n_bins=50, max_dist=None):
        """Computes and fits semivariograms and a cross-semivariograms via composite WLS.

        NOTE: Kernels could be updated with fitted parameters.
        """
        variograms, covariograms, params = vg.variogram_analysis(
            self.fields, params_guess, n_bins=n_bins, max_dist=max_dist
        )
        self.fields.variograms = variograms
        self.fields.covariograms = covariograms

        # params_arr = np.hstack([params[name] for name in names])
        # self.set_params(params_arr)
        return variograms, covariograms, params

    # def neg_log_lik(self, params, dist_blocks):
    #     """Computes the (negative) log-likelihood of the supplied parameters."""
    #     # construct joint covariance matrix
    #     self.set_params(params)
    #     cov_mat = self.covariance_matrix(dist_blocks)

    #     # inverse and determinant via Cholesky decomposition
    #     try:
    #         cho_l, low = cho_factor(cov_mat, lower=True)
    #     except:
    #         # covariance matrix is not positive definite
    #         return np.inf

    #     log_det = np.sum(np.log(np.diag(cho_l)))
    #     quad_form = np.matmul(
    #         self.fields.joint_data_vec,
    #         cho_solve((cho_l, low), self.fields.joint_data_vec),
    #     )

    #     # negative log-likelihood (up to normalizing constants)
    #     return log_det + 0.5 * quad_form

    # def fit(self, initial_guess=None, options=None):
    #     """Fit model parameters by maximum likelihood estimation."""
    #     # TODO: allow for variable smoothness, maybe use exponential transform for all but rho
    #     if initial_guess is None:
    #         initial_guess = list(self.get_params().values())
    #     bounds = list(self.param_bounds.values())
    #     dist_blocks = self.fields.get_joint_dists()

    #     # minimize the negative log-likelihood
    #     optim_res = minimize(
    #         self.neg_log_lik,
    #         initial_guess,
    #         bounds=bounds,
    #         args=(dist_blocks),
    #         method="L-BFGS-B",
    #         options=options,
    #     )
    #     self.set_params(optim_res.x)
    #     if optim_res.success is not True:
    #         raise Exception(
    #             "ERROR: optimization did not converge. Terminated with message:"
    #             f" {optim_res.message}"
    #         )
    #     # check parameter validity (Gneiting et al. 2010, or just psd check?)
    #     # NOTE: this happens (the correct way) in cokrige.call(); should we do it here too?
    #     # cho_factor(self.covariance_matrix(dist_blocks))
    #     return self


def matern_correlation(h: np.ndarray, nu: float, len_scale: float):
    r"""Matern correlation function.

    Parameters:
        h: array of spatial separation distances (lags)
        nu, len_scale: see MaternCovariance

    .. math::
        \rho(h) =
        \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
        \left(\sqrt{2\nu}\cdot\frac{h}{\ell}\right)^{\nu} \cdot
        \mathrm{K}_{\nu}\left(\sqrt{2\nu}\cdot\frac{h}{\ell}\right)
    """
    # TODO: add check so that negative distances and correlation values yeild warning.
    h = np.array(np.abs(h), dtype=np.double)
    # calculate by log-transformation to prevent numerical errors
    h_positive_scaled = h[h > 0.0] / len_scale
    corr = np.ones_like(h)
    corr[h > 0.0] = np.exp(
        (1.0 - nu) * np.log(2)
        - sps.gammaln(nu)
        + nu * np.log(np.sqrt(2.0 * nu) * h_positive_scaled)
    ) * sps.kv(nu, np.sqrt(2.0 * nu) * h_positive_scaled)
    # if nu >> 1 we get errors for the farfield, there 0 is approached
    corr[np.logical_not(np.isfinite(corr))] = 0.0
    # Matern correlation is positive
    corr = np.maximum(corr, 0.0)
    return corr
