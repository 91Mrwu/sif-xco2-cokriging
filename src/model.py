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

    def count_params(self):
        return len(self.get_values())

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
        rho: co-located cross-correlation coefficient(s)
    """

    def __init__(self, n_procs: int = 2) -> None:
        self.n_procs = n_procs
        self.sigma = MarginalParam("sigma", 1.0, (0.4, 3.5), dim=n_procs)
        self.nu = CrossParam("nu", 1.5, (0.2, 3.5), dim=n_procs)
        self.len_scale = CrossParam("len_scale", 5e2, (1e2, 2e3), dim=n_procs)
        self.nugget = MarginalParam("nugget", 0.0, (0.0, 0.2), dim=n_procs)
        self.rho = RhoParam("rho", 0.0, (-1.0, 1.0), dim=n_procs)
        self._params = [self.sigma, self.nu, self.len_scale, self.nugget, self.rho]
        self.n_params = 0
        for p in self._params:
            self.n_params += p.count_params()

    def to_dataframe(self):
        df_list = [p.to_dataframe() for p in self._params]
        return pd.concat(df_list, ignore_index=True)

    def get_values(self):
        return self.to_dataframe()["value"].values

    def set_values(self, x: np.ndarray):
        if len(x) != self.n_params:
            raise ValueError("Incorrect number of parameters in input array.")
        for p in self._params:
            n_params = p.count_params()
            vals, x = x[:n_params], x[n_params:]
            p.set_values(vals)
        return self

    def get_bounds(self):
        return self.to_dataframe()["bounds"].values

    def set_bounds(self, **kwargs):
        for name, bounds in kwargs.items():
            try:
                param = getattr(self, name)
                param.bounds = bounds
            except AttributeError:
                raise AttributeError(f"`{name}` is not a valid parameter.")
        return self


# TODO: add method to check parameter validity (overall pos def and paper specs)
# NOTE: should this be a subclass of MaternParams?
class FullBivariateMatern:
    """Full bivariate Matern covariance model (Gneiting et al., 2010).

    Notes:
    - Parameterization follows Rassmussen and Williams (2006; see MaternParams)
    """

    def __init__(self):
        self.params = MaternParams(n_procs=2)

    def correlation(self, h: np.ndarray):
        return matern_correlation(h, *self.params[[1, 2]])

    def covariance(self, i: int, j: int, h: np.ndarray):
        """Matern covariance function."""
        # can use, e.g. self.params.sigma.values[i, j]
        cov = self.params[0] ** 2 * self.correlation(h)
        cov[h == 0] += self.params[3]
        return cov

    def cross_covariance(self, h: np.ndarray):
        return (
            self.params.rho
            * self.params.sigmas[0]
            * self.params.sigmas[1]
            * self.correlation(h)
        )

    def semivariance(self):
        pass

    def cross_semivariance(self):
        pass

    # prediction stuff

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
