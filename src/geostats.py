## Classes for cokriging, covariance kernels, etc.
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr

import scipy.special as sps
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

    def fit_model(self, kernel):
        pass

    def _get_time_lag(self, timestamp, timedelta):
        t0 = datetime.strptime(timestamp, "%Y-%m-%d")
        return (t0 - relativedelta(months=timedelta)).strftime("%Y-%m-%d")


class Kernel(MultiField):
    """
    Gaussian process kernel.
    """

    def __init__(self):
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
        return {
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


class Cokrige(MultiField):
    """
    Details and references
    """

    def __init__(self, field_1, field_2, model, dist_units="km", fast_dist=False):
        super().__init__(field_1, field_2)
        self.model = model
        self.dist_units = dist_units
        self.fast_dist = fast_dist

    def __call__(self, loc):
        # compute the kriging covariance matrix
        cov_mat = self._get_cov_mat()

        # return joint covariance matrix

        # return self.pred, self.uncertainty
        pass

    def _get_dists(self):
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

    def _get_cov_mat(self):
        """Computes the cokriging joint covariance matrix."""
        # collect block covariance matrices
        dist_mats = self._get_dists()
        cov_mats = {}
        for block, dist_mat in dist_mats.items():
            cov_mats[block] = self.model.cov_nugget(dist_mat)

        # stack blocks into joint covariance matrix
        return np.block(
            [
                [cov_mats["block_11"], cov_mats["block_12"]],
                [cov_mats["block_21"], cov_mats["block_22"]],
            ]
        )

