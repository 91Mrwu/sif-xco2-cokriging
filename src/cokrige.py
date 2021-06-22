import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.interpolate import griddata

import spatial_tools
import cov_model


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

