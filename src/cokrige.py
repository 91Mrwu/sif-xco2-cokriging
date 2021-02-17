import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
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

    def __call__(self, pred_loc, full_cov=False):
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

        ## Check model validity
        try:
            self._check_pos_def(Sigma_11, Sigma_12, Sigma_22)
        except LinAlgError:
            print("The joint covariance matrix is not positive definite.")
            raise

        ## Prediction
        # TODO: refactor to seperate function to handle mean, trend, transforms, etc.
        mu = self._get_pred_mean(pred_loc, ds=self.fields.ds_1)
        sigma = self._get_pred_std(pred_loc, ds=self.fields.ds_1)
        self.pred = mu + sigma * np.matmul(
            Sigma_12, cho_solve(cho_factor(Sigma_22, lower=True), Z)
        )

        ## Prediction covariance and error
        raw_cov_mat = Sigma_11 - np.matmul(
            Sigma_12, cho_solve(cho_factor(Sigma_22, lower=True), Sigma_12.T)
        )
        self.pred_cov = krige_tools.pre_post_diag(sigma, raw_cov_mat)
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

    def _get_pred_mean(self, pred_loc, method="temporal", ds=None):
        """Fits the surface mean at prediction locations using the supplied mean function.
        
        TODO: make mean function more flexible.
        """
        if method == "temporal":
            df = (
                krige_tools.preprocess_ds(ds)
                .sel(time=self.fields.field_1.timestamp)
                .to_dataframe()
                .reset_index()
                .dropna(subset=["mean"])
            )
            coords = df[["lat", "lon"]].values
            field_mean = df["mean"].values
        else:
            print(f"Error: method '{method}' not implemented.")
        return griddata(coords, field_mean, pred_loc, method="nearest")

    def _get_pred_std(self, pred_loc, method="temporal", ds=None):
        """Fits the surface standard deviation at prediction locations using the supplied mean function."""
        if method == "temporal":
            df = (
                krige_tools.preprocess_ds(ds)
                .sel(time=self.fields.field_1.timestamp)
                .to_dataframe()
                .reset_index()
                .dropna(subset=["std"])
            )
            coords = df[["lat", "lon"]].values
            field_std = df["std"].values
        else:
            print(f"Error: method '{method}' not implemented.")
        return griddata(coords, field_std, pred_loc, method="nearest")

