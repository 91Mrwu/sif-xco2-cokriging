import warnings

from numba import njit, prange
import numpy as np

# import gstools as gs
import scipy.special as sps
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.optimize import minimize

import krige_tools
import variogram as vgm


class Matern:
    """The Matern covariance model.
    
    TODO: make this a subclass of gs.CovModel?
    """

    def __init__(self, sigma=1.0, nu=1.5, len_scale=1.0, nugget=0.0):
        self.sigma = sigma  # process standard deviation
        self.nu = nu  # smoothess parameter
        self.len_scale = len_scale  # length scale parameter
        self.nugget = nugget  # nugget parameter (NOTE: already squared)

    def correlation(self, h):
        r"""Matérn correlation function.

        .. math::
           \rho(r) =
           \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
           \left(\sqrt{2\nu}\cdot\frac{r}{\ell}\right)^{\nu} \cdot
           \mathrm{K}_{\nu}\left(\sqrt{2\nu}\cdot\frac{r}{\ell}\right)
        """
        # NOTE: modified version of gptools.CovModel.Matern.correlation
        # TODO: add check so that negative distances and correlation values yeild warning.
        h = np.array(np.abs(h), dtype=np.double)
        # calculate by log-transformation to prevent numerical errors
        h_gz = h[h > 0.0]
        res = np.ones_like(h)
        res[h > 0.0] = np.exp(
            (1.0 - self.nu) * np.log(2)
            - sps.gammaln(self.nu)
            + self.nu * np.log(np.sqrt(2.0 * self.nu) * h_gz / self.len_scale)
        ) * sps.kv(self.nu, np.sqrt(2.0 * self.nu) * h_gz / self.len_scale)
        # if nu >> 1 we get errors for the farfield, there 0 is approached
        res[np.logical_not(np.isfinite(res))] = 0.0
        # covariance is positive
        res = np.maximum(res, 0.0)
        return res


class BivariateMatern:
    """Bivariate Matern kernel, or correlation function."""

    def __init__(
        self, fields, kernel_1, kernel_2, rho=0.0, nu_b=None, len_scale_b=None
    ):
        self.rho = rho  # co-located correlation coefficient
        if nu_b is None:
            nu_b = 0.5 * (kernel_1.nu + kernel_2.nu)
        if len_scale_b is None:
            len_scale_b = 0.5 * (kernel_1.len_scale + kernel_2.len_scale)
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.kernel_b = Matern(nu=nu_b, len_scale=len_scale_b)

        self.fields = fields
        # NOTE: do we want to take the mean here, or add values along diagonal individually?
        self.sigep_11 = fields.field_1.variance_estimate.mean()
        self.sigep_22 = fields.field_2.variance_estimate.mean()

        # reasonable
        # self.param_bounds = {
        #     "sigma_11": (0.1, 0.4),
        #     # "nu_11": [0.2, 5.0],
        #     "len_scale_11": (5e2, 2e3),
        #     "nugget_11": (0.0, 1.0),
        #     # "nu_12": [0.2, 5.0],
        #     "len_scale_12": (1e3, 5e3),
        #     "rho": (-1.0, -0.01),
        #     "sigma_22": (0.7, 1.2),
        #     # "nu_22": [0.2, 5.0],
        #     "len_scale_22": (5e2, 2e3),
        #     "nugget_22": (0.0, 1.0),
        # }
        # restricted
        self.param_bounds = {
            "sigma_11": (0.2, 4.0),
            "nu_11": [0.2, 0.5],
            "len_scale_11": (2e3, 5e4),
            "nugget_11": (0.0, 1.0),
            "nu_12": [0.2, 0.5],
            "len_scale_12": (2e3, 5e4),
            "rho": (-1.0, -0.01),
            "sigma_22": (0.2, 4.0),
            "nu_22": [0.2, 0.5],
            "len_scale_22": (2e3, 5e4),
            "nugget_22": (0.0, 1.0),
        }

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
        return krige_tools.pre_post_diag(self.fields.joint_std_inverse, cov_mat)

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

    # def _params_from_gstools(
    #     self, field, bin_edges, sampling_size=None, sampling_seed=None
    # ):
    #     """
    #     NOTE: This variogram model is represented over Euclidean distance so lengh scale will be wrong (though haversine distance availalbe via gs.variogram.estimator.unstructured). If we want to make a variogram estimate available, we should write it ourselves (then we also have more control, e.g. warnings when there are less than 30 obs in a bin).
    #     """
    #     # estimate variogram
    #     bin_center, gamma = gs.vario_estimate_unstructured(
    #         (field.coords[:, 1], field.coords[:, 0]),
    #         field.values,
    #         bin_edges,
    #         sampling_size=sampling_size,
    #         sampling_seed=sampling_seed,
    #         estimator="cressie",
    #     )
    #     # fit a Matern variogram model
    #     # NOTE: may want to use custom Matern formulation
    #     fit_model = gs.Matern(dim=2, nu=2.5)
    #     fit_model.set_arg_bounds(var=[0.1, 10], nu=[0.2, 5], len_scale=[1, 500])
    #     params, _ = fit_model.fit_variogram(bin_center, gamma, nu=False, nugget=False)
    #     return (
    #         params,
    #         {"model": fit_model, "bins": bin_center, "emp_semivariogram": gamma},
    #     )

    # def _gs_kernels(self, bin_edges, sampling_size=None, sampling_seed=None):
    #     """
    #     Collects parameters needed for construction of process kernels and cross-kernels.

    #     TODO: add ability to set each parameter; special feature in gstools for individual kernels, and manual for cross kernels.
    #     """
    #     params_1, vario_obj1 = self._params_from_variogram(
    #         self.fields.field_1,
    #         bin_edges,
    #         sampling_size=sampling_size,
    #         sampling_seed=sampling_seed,
    #     )
    #     params_2, vario_obj2 = self._params_from_variogram(
    #         self.fields.field_2,
    #         bin_edges,
    #         sampling_size=sampling_size,
    #         sampling_seed=sampling_seed,
    #     )

    #     self.kernel_1 = Matern(
    #         sigma=np.sqrt(params_1["var"]),
    #         nu=params_1["nu"],
    #         len_scale=params_1["len_scale"],
    #         nugget=params_1["nugget"],
    #     )
    #     self.kernel_2 = Matern(
    #         sigma=np.sqrt(params_2["var"]),
    #         nu=params_2["nu"],
    #         len_scale=params_2["len_scale"],
    #         nugget=params_2["nugget"],
    #     )
    #     self.kernel_b = Matern(
    #         nu=0.5 * (self.kernel_1.nu + self.kernel_2.nu),
    #         len_scale=0.5 * (self.kernel_1.len_scale + self.kernel_2.len_scale),
    #     )
    #     self.rho = pearsonr(
    #         *krige_tools.match_data_locations(self.fields.field_1, self.fields.field_2)
    #     )[0]

    #     return self, (vario_obj1, vario_obj2)

    def empirical_variograms(
        self,
        cov_guesses,
        cross_guess,
        n_bins=15,
        standardize=False,
        shift_coords=False,
    ):
        """Computes and fits individual variograms and a cross-covariogram. Kernels are updated with fitted parameters."""
        variograms, covariograms, params = vgm.variogram_analysis(
            self.fields,
            cov_guesses,
            cross_guess,
            n_bins=n_bins,
            standardize=standardize,
            shift_coords=shift_coords,
        )
        self.fields.variograms = variograms
        self.fields.covariograms = covariograms

        # params_arr = np.hstack([params[name] for name in names])
        # self.set_params(params_arr)
        return variograms, covariograms, params

    def neg_log_lik(self, params, dist_blocks):
        """Computes the (negative) log-likelihood of the supplied parameters."""
        # construct joint covariance matrix
        self.set_params(params)
        cov_mat = self.covariance_matrix(dist_blocks)

        # inverse and determinant via Cholesky decomposition
        try:
            cho_l, low = cho_factor(cov_mat, lower=True)
        except:
            # covariance matrix is not positive definite
            return np.inf

        log_det = np.sum(np.log(np.diag(cho_l)))
        quad_form = np.matmul(
            self.fields.joint_data_vec,
            cho_solve((cho_l, low), self.fields.joint_data_vec),
        )

        # negative log-likelihood (up to normalizing constants)
        return log_det + 0.5 * quad_form

    def fit(self, initial_guess=None, options=None):
        """Fit model parameters by maximum likelihood estimation."""
        # TODO: allow for variable smoothness, maybe use exponential transform for all but rho
        if initial_guess is None:
            initial_guess = list(self.get_params().values())
        bounds = list(self.param_bounds.values())
        dist_blocks = self.fields.get_joint_dists()

        # minimize the negative log-likelihood
        optim_res = minimize(
            self.neg_log_lik,
            initial_guess,
            bounds=bounds,
            args=(dist_blocks),
            method="L-BFGS-B",
            options=options,
        )
        self.set_params(optim_res.x)
        if optim_res.success is not True:
            raise Exception(
                f"ERROR: optimization did not converge. Terminated with message: {optim_res.message}"
            )
        # check parameter validity (Gneiting et al. 2010, or just psd check?)
        # NOTE: this happens (the correct way) in cokrige.call(); should we do it here too?
        # cho_factor(self.covariance_matrix(dist_blocks))
        return self
