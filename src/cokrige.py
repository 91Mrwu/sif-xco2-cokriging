import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.interpolate import griddata

import krige_tools
import fields
import cov_model


class Cokrige:
    """
    Details and references

    TODO:
    - remove field means and/or standardize data
    """

    def __init__(self, fields, model, dist_units="km", fast_dist=False):
        self.fields = fields
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
        # data vector
        Z = np.hstack((self.fields.field_1.values, self.fields.field_2.values))

        ## Prediction
        # TODO: refactor to seperate function to handle mean, trend, transforms, etc.
        self.pred = self._get_pred_mean(pred_loc) + np.matmul(
            Sigma_12, cho_solve(cho_factor(Sigma_22, lower=True), Z)
        )

        ## Standard error
        self.pred_cov = Sigma_11 - np.matmul(
            Sigma_12, cho_solve(cho_factor(Sigma_22, lower=True), Sigma_12.T)
        )
        # TODO: handle cases with negative entries appropriately
        self.pred_error = np.sqrt(np.abs(np.diagonal(self.pred_cov)))
        # self.pred_error = np.sqrt(np.diagonal(self.pred_cov))

        return self.pred, self.pred_error

    def _get_pred_dist(self, pred_loc):
        """Computes distances between prediction point(s)."""
        return krige_tools.distance_matrix(
            pred_loc, pred_loc, units=self.dist_units, fast_dist=self.fast_dist
        )

    def _get_cross_dist(self, pred_loc):
        """Computes distances between prediction point(s) and data values."""
        return {
            "block_11": krige_tools.distance_matrix(
                pred_loc,
                self.fields.field_1.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
            "block_12": krige_tools.distance_matrix(
                pred_loc,
                self.fields.field_2.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
        }

    def _get_joint_dists(self):
        """Computes block distance matrices and returns the blocks in a dict."""
        off_diag = krige_tools.distance_matrix(
            self.fields.field_1.coords,
            self.fields.field_2.coords,
            units=self.dist_units,
            fast_dist=self.fast_dist,
        )
        return {
            "block_11": krige_tools.distance_matrix(
                self.fields.field_1.coords,
                self.fields.field_1.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
            "block_12": off_diag,
            "block_21": off_diag.T,
            "block_22": krige_tools.distance_matrix(
                self.fields.field_2.coords,
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
        return self.model.covariance_matrix(self._get_joint_dists())

    def _get_pred_mean(self, pred_loc):
        """Fits the surface mean at prediction locations using the supplied mean function.
        
        TODO: make mean function more flexible.
        """
        return griddata(
            self.fields.field_1.coords,
            self.fields.field_1.mean,
            pred_loc,
            method="nearest",
        )

