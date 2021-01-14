## Classes for cokriging, covariance kernels, etc.
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr

import scipy.special as sps
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances

import gstools as gs


def expand_grid(self, *args):
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
    # enfore 2d array in case single point
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


class Field:
    """
    Stores data values and coordinates for a single process at a fixed time in a data frame.
    """

    def __init__(self, da, timestamp):
        df = da.sel(time=timestamp).to_dataframe().reset_index()
        self.timestamp = datetime.strptime(timestamp, "%Y-%m-%d")
        self.coords = df[["lat", "lon"]].values
        self.values = df.values[:, -1]


class MultiField:
    """
    Main class; home for data and predictions.
    """

    def __init__(self, field_1, field_2, normalize_values=False):
        # TODO: implement normalize values
        self.field_1 = field_1
        self.field_2 = field_2
        self.timestamp = field_1.timestamp
        # self.timedelta = field_1.timestamp - field_2.timestamp
        self.normalize_values = normalize_values

    def _get_time_lag(self, timestamp, timedelta):
        """Time lag"""
        t0 = datetime.strptime(timestamp, "%Y-%m-%d")
        return (t0 - relativedelta(months=timedelta)).strftime("%Y-%m-%d")


class Matern:
    """The Matern covariance model."""

    def __init__(self, sigma=1.0, nu=1.0, len_scale=1.0, nugget=0.0):
        self.sigma = sigma  # process standard deviation
        self.nu = nu  # smoothess parameter
        self.len_scale = len_scale  # length scale parameter
        self.nugget = nugget  # nugget parameter

    def correlation(self, h):
        r"""Matérn correlation function.

        .. math::
           \rho(r) =
           \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
           \left(\sqrt{\nu}\cdot\frac{r}{\ell}\right)^{\nu} \cdot
           \mathrm{K}_{\nu}\left(\sqrt{\nu}\cdot\frac{r}{\ell}\right)
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
            + self.nu * np.log(np.sqrt(self.nu) * h_gz / self.len_scale)
        ) * sps.kv(self.nu, np.sqrt(self.nu) * h_gz / self.len_scale)
        # if nu >> 1 we get errors for the farfield, there 0 is approached
        res[np.logical_not(np.isfinite(res))] = 0.0
        # covariance is positive
        res = np.maximum(res, 0.0)
        return res


class BivariateMatern:
    """Bivariate Matern kernel, or correlation function"""

    def __init__(self, kernel_1, kernel_2, rho=1.0, nu_b=1.0, len_scale_b=1.0):
        self.rho = rho  # co-located correlation coefficient
        self.nu_b = nu_b  # joint smoothness
        self.len_scale_b = len_scale_b  # joint length scale parameter
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.kernel_b = Matern(nu=nu_b, len_scale=len_scale_b)

    def pred_covariance(self, dist_mat):
        """Computes the variance-covariance matrix for prediction location(s)."""
        return (
            self.kernel_1.sigma ** 2 * self.kernel_1.correlation(dist_mat)
            + self.kernel_1.nugget
        )

    def cross_covariance(self, dist_blocks):
        """Computes the cross-covariance vectors for prediction distances."""
        c_11 = self.kernel_1.sigma ** 2 * self.kernel_1.correlation(
            dist_blocks["block_11"]
        )
        c_12 = self.kernel_2.sigma ** 2 * self.kernel_2.correlation(
            dist_blocks["block_12"]
        )
        return np.hstack((c_11, c_12))

    def covariance_matrix(self, dist_blocks):
        """Constructs the bivariate Matern covariance matrix."""
        ## TODO: add measurement error term for C_11 and C_22
        C_11 = (
            self.kernel_1.sigma ** 2
            * self.kernel_1.correlation(dist_blocks["block_11"])
            + self.kernel_1.nugget
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
        C_22 = (
            self.kernel_2.sigma ** 2
            * self.kernel_2.correlation(dist_blocks["block_22"])
            + self.kernel_2.nugget
        )
        # stack blocks into joint covariance matrix
        return np.block([[C_11, C_12], [C_21, C_22]])

    def fit(self):
        pass

    def _params_from_variogram(
        self, field, bin_edges, sampling_size=None, sampling_seed=None
    ):
        # estimate variogram
        bin_center, gamma = gs.vario_estimate_unstructured(
            (field.coords[:, 0], field.coords[:, 1]),
            field.values,
            bin_edges,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
            estimator="cressie",
        )
        # fit a Matern variogram model
        fit_model = gs.Matern(dim=2)  # NOTE: may want to use custom Matern formulation
        params, _ = fit_model.fit_variogram(bin_center, gamma)
        return params

    def _empirical_params(
        self, F1, F2, bin_edges, sampling_size=None, sampling_seed=None
    ):
        """
        Collects parameters needed for construction of process kernels and cross-kernels.
        """
        params_1 = self._params_from_variogram(
            F1, bin_edges, sampling_size=None, sampling_seed=None
        )
        params_2 = self._params_from_variogram(
            F2, bin_edges, sampling_size=None, sampling_seed=None
        )
        self.params = {
            "var_1": params_1["var"],
            "len_scale_1": params_1["len_scale"],
            "nugget_1": params_1["nugget"],
            "nu_1": params_1["nu"],
            "var_2": params_2["var"],
            "len_scale_2": params_2["len_scale"],
            "nugget_2": params_2["nugget"],
            "nu_2": params_2["nu"],
            "rho": np.corrcoef(F1.values, F2.values)[0, 1],
        }
        return self.params


class Cokrige(MultiField):
    """
    Details and references
    """

    def __init__(self, field_1, field_2, model, dist_units="km", fast_dist=False):
        super().__init__(field_1, field_2)
        self.model = model
        self.dist_units = dist_units
        self.fast_dist = fast_dist

    def __call__(self, pred_loc):
        """Cokriging predictor and prediction standard error.

        NOTE: covariance matrix inversion via Cholesky decomposition
        """
        # prediction variance-covariance matrix
        Sigma_11 = self._get_pred_cov(pred_loc)
        # prediction cross-covariance vectors
        Sigma_12 = self._get_cross_cov(pred_loc)
        # cokriging joint covariance matrix
        Sigma_22 = self._get_joint_cov()
        # collect data values
        Z = np.hstack((self.field_1.values, self.field_2.values))

        ## Prediction
        self.pred = np.matmul(Sigma_12, cho_solve(cho_factor(Sigma_22, lower=True), Z))

        ## Variance
        self.pred_err = Sigma_11 - np.matmul(
            Sigma_12, cho_solve(cho_factor(Sigma_22, lower=True), Sigma_12.T)
        )

        return self.pred, self.pred_err

    def _get_pred_dist(self, pred_loc):
        """Computes distances between prediction point(s)."""
        return distance_matrix(
            pred_loc, pred_loc, units=self.dist_units, fast_dist=self.fast_dist
        )

    def _get_cross_dist(self, pred_loc):
        """Computes distances between prediction point(s) and data values."""
        return {
            "block_11": distance_matrix(
                pred_loc,
                self.field_1.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
            "block_12": distance_matrix(
                pred_loc,
                self.field_2.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
        }

    def _get_joint_dists(self):
        """Computes block distance matrices and returns the blocks in a dict."""
        off_diag = distance_matrix(
            self.field_1.coords,
            self.field_2.coords,
            units=self.dist_units,
            fast_dist=self.fast_dist,
        )
        return {
            "block_11": distance_matrix(
                self.field_1.coords,
                self.field_1.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
            "block_12": off_diag,
            "block_21": off_diag.T,
            "block_22": distance_matrix(
                self.field_1.coords,
                self.field_1.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
        }

    def _get_pred_cov(self, pred_loc):
        """Computes the variance-covariance matrix for prediction locations."""
        return self.model.pred_covariance(self._get_pred_dist(pred_loc))

    def _get_cross_cov(self, pred_loc):
        """Computes the cross-covariance vectors for prediction locations."""
        return self.model.cross_covariance(self._get_cross_dist(pred_loc))

    def _get_joint_cov(self):
        """Computes the cokriging joint covariance matrix."""
        return self.model.covariance_matrix(self._get_joint_dists())

    def _kriging_predictor(self):
        """Pred"""
        pass

    def _kriging_variance(self):
        """Var"""
        pass
