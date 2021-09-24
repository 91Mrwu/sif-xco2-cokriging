"""
Steps:
[x] recover the data values along the main grid points only (i.e., original 4x5-degree grid)
[x] setup prediction locations with resultion 0.5x0.5-degree (land only)
[x] precompute the grand-covariance matrix blocks for all data locations
[x] for a given prediction location, find the data indices within a max_dist for each dataset
    3.1 if there are more than N points in that window, then subsample [rather than subsample, shrink the window]
[x] collect the block covariance matrix and the stacked data vector at the correct indices
6. compute covariance arrays between prediction locations and data locations
7. compute prediction of residual process
8. compute the uncertainty for the residuals
9. post-process the predicted residuals
    9.1. predict the mean surface at prediction (standardized) locations or (standardized) evi [need to update EVI dataset for this]
    9.2. multiply by the scale factor and add the (constant) spatial mean
    9.3. add the spatial trend surface
    9.4. add the temporal trend value
    9.5. multiply diagonal elements of uncertainty value by scale factor squared, then take the square root of that value (gives pred uncertainty)

TODO: switch from dict ref to nested list ref in blocks
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

import data_utils
import spatial_tools
from fields import MultiField
from model import FullBivariateMatern


@dataclass
class PredConfig:
    """Multivariate spatial prediction configuration."""

    pcoords: np.ndarray
    max_dist: float
    dist_units: str = "km"
    fast_dist: bool = True
    n_max: int = None


class Pred:
    """Multivariate prediction framework."""

    def __init__(
        self, mod: FullBivariateMatern, mf: MultiField, config: PredConfig
    ) -> None:
        if mod.n_procs != mf.n_procs:
            raise ValueError(
                "Number of theoretical processes different from empirical processes."
            )
        self.n_procs = mod.n_procs
        self.config = config
        self.mod = mod
        self.mf = mf
        self.Sigma = self._cov_blocks()

    def _cov_blocks(self):
        """Precomputes each block in the block-covariance matrix, with each block describing the dependence within a process or between processes."""
        # np.nan * np.zeros((self.mf.fields[i].size, self.mf.fields[j].size))
        blocks = dict()
        for i in range(self.n_procs):
            for j in range(self.n_procs):
                if i <= j:
                    h = self.mf.calc_dist_matrix(
                        (i, j), self.config.dist_units, self.config.fast_dist, main=True
                    )
                    if i == j:
                        blocks[f"{i}{j}"] = self.mod.covariance(i, h)
                        # add measurement error variance along diagonals
                        # np.fill_diagonal(blocks[i, j], blocks[i, j].diagonal() + sigep[i])
                    else:
                        blocks[f"{i}{j}"] = self.mod.cross_covariance(i, j, h)

        # NOTE: is normalization by data std necessary?
        # stack blocks into joint covariance matrix and normalize by standard deviation
        # cov_mat = np.block([[C_11, C_12], [C_21, C_22]])
        # return pre_post_diag(self.fields.joint_std_inverse, cov_mat)
        return blocks

    def _local_ix(self, s0: np.ndarray, max_dist: float, n_max: int = None):
        """Determines the indices of all data locations within `max_dist` of the specified prediction location `s0` for each dataset."""
        dists = [
            spatial_tools.distance_matrix(
                s0,
                f.coords_main,
                units=self.config.dist_units,
                fast_dist=self.config.fast_dist,
            )
            for f in self.mf.fields
        ]
        # NOTE: may need to adjust max_dist if the size of any of these elements is larger than n_max
        ix = [(d < max_dist).squeeze() for d in dists]
        return ix

    def _local_values(self, s0: np.ndarray, max_dist: float, n_max: int = None) -> list:
        """For a given prediction location s0, return the data vector and covariance matrix based on locations within `max_dist`."""
        ix = self._local_ix(s0, max_dist, n_max)
        local_data = [self.mf.fields[i].values_main[ix[i]] for i in range(self.n_procs)]
        local_cov_blocks = dict()
        for i in range(self.n_procs):
            for j in range(self.n_procs):
                if i <= j:
                    local_cov_blocks[f"{i}{j}"] = self.Sigma[f"{i}{j}"][
                        np.ix_(ix[i], ix[j])
                    ]
                else:
                    local_cov_blocks[f"{i}{j}"] = self.Sigma[f"{j}{i}"][
                        np.ix_(ix[j], ix[i])
                    ].T
        # format the data as a vector and the covariance blocks as a matrix
        local_data = np.hstack(local_data)
        local_cov = np.block(
            [
                [local_cov_blocks[f"{i}{j}"] for j in range(self.n_procs)]
                for i in range(self.n_procs)
            ]
        )
        assert local_data.shape[0] == local_cov.shape[0]
        return local_data, local_cov

    #######

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


class Cokrige:
    """
    Details and references.
    """

    def __init__(self, fields, model):
        self.fields = fields
        self.model = model
        self.dist_units = fields.dist_units
        self.fast_dist = fields.fast_dist

    def __call__(self, pred_loc, full_cov=False):
        """
        Cokriging predictor and prediction standard error.
        """
        # prediction variance-covariance matrix
        Sigma_11 = self._get_pred_cov(pred_loc)
        # prediction cross-covariance vectors
        Sigma_12 = self._get_cross_cov(pred_loc)
        # cokriging joint covariance matrix
        Sigma_22 = self._get_joint_cov()

        ## Check model validity
        try:
            self._check_pos_def(Sigma_11, Sigma_12, Sigma_22)
        except LinAlgError:
            warnings.warn("The joint covariance matrix is not positive definite.")

        ## Prediction
        # TODO: refactor to seperate function to handle mean, trend, transforms, etc.
        # TODO: need to reconsider how we want to get mean, std, trend (annual window)
        mu = self._get_pred_mean(pred_loc, ds=self.fields.ds_1)
        sigma = self._get_pred_std(pred_loc, ds=self.fields.ds_1)
        self.pred = mu + sigma * np.matmul(
            Sigma_12,
            cho_solve(cho_factor(Sigma_22, lower=True), self.fields.joint_data_vec),
        )

        ## Prediction covariance and error
        norm_cov_mat = Sigma_11 - np.matmul(
            Sigma_12, cho_solve(cho_factor(Sigma_22, lower=True), Sigma_12.T)
        )
        self.pred_cov = spatial_tools.pre_post_diag(sigma, norm_cov_mat)
        self.pred_error = np.sqrt(np.diagonal(self.pred_cov))

        if full_cov:
            return self.pred, self.pred_cov
        else:
            return self.pred, self.pred_error

    def _check_pos_def(self, Sig11, Sig12, Sig22, Sig21=None):
        """Check that the overarching joint covariance matrix is positive definite using the Cholesky decompostion."""
        if Sig21 is None:
            Sig21 = Sig12.T
        cho_factor(np.block([[Sig11, Sig12], [Sig21, Sig22]]))

    def _get_pred_dist(self, pred_loc):
        """Computes distances between prediction point(s)."""
        return spatial_tools.distance_matrix(
            pred_loc, pred_loc, units=self.dist_units, fast_dist=self.fast_dist
        )

    def _get_cross_dist(self, pred_loc):
        """Computes distances between prediction point(s) and data values."""
        return {
            "block_11": spatial_tools.distance_matrix(
                pred_loc,
                self.fields.field_1.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
            "block_12": spatial_tools.distance_matrix(
                pred_loc,
                self.fields.field_2.coords,
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
        return self.model.covariance_matrix(self.fields.get_joint_dists())

    def _get_pred_mean(self, pred_loc, method="temporal", ds=None):
        """Fits the surface mean at prediction locations using the supplied mean function.

        TODO: make mean function more flexible.
        """
        if method == "temporal":
            df = (
                spatial_tools.preprocess_ds(ds)
                .sel(time=self.fields.field_1.timestamp)
                .to_dataframe()
                .reset_index()
                .dropna(subset=["mean"])
            )
            coords = df[["lat", "lon"]].values
            field_mean = df["mean"].values
        else:
            warnings.warn(f"Error: method '{method}' not implemented.")
        return griddata(coords, field_mean, pred_loc, method="nearest")

    def _get_pred_std(self, pred_loc, method="temporal", ds=None):
        """Fits the surface standard deviation at prediction locations using the supplied mean function."""
        if method == "temporal":
            df = (
                spatial_tools.preprocess_ds(ds)
                .sel(time=self.fields.field_1.timestamp)
                .to_dataframe()
                .reset_index()
                .dropna(subset=["std"])
            )
            coords = df[["lat", "lon"]].values
            field_std = df["std"].values
        else:
            warnings.warn(f"Error: method '{method}' not implemented.")
        return griddata(coords, field_std, pred_loc, method="nearest")


def prediction_coords(
    extents: tuple = (-125, -65, 22, 58), lon_res: float = 0.5, lat_res: float = 0.5
) -> np.ndarray:
    """Produces prediction coordinates (land only)."""
    grid = data_utils.GridConfig(extents=extents, lon_res=lon_res, lat_res=lat_res)
    df = data_utils.land_grid(grid)
    return df.reset_index()[["lat", "lon"]].values
