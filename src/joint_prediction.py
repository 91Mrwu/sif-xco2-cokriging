import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from fields import MultiField, distance_matrix
from model import MultivariateMatern
from data_utils import GridConfig, land_grid


class Predictor:
    """Multivariate prediction framework."""

    def __init__(
        self,
        mod: MultivariateMatern,
        mf: MultiField,
        covariates: xr.DataArray = None,
        dist_units: str = "km",
        fast_dist: bool = True,
    ) -> None:
        if mod.n_procs != mf.n_procs:
            raise ValueError(
                "Number of theoretical processes different from empirical processes."
            )
        self.n_procs = mod.n_procs
        self.mod = mod
        self.mf = mf
        self.covariates = covariates
        self.dist_units = dist_units
        self.fast_dist = fast_dist

    def __call__(
        self, i: int, pcoords: pd.DataFrame, postprocess: bool = True, cv_ix: int = None
    ) -> pd.DataFrame:
        """Apply the multivariate local prediction at each location in the specified prediction coordinates.

        Parameters:
            i: index of the process to be predicted
            pcoords: prediction coordinates with format [[lat, lon]]
            postprocess: should the `mf` and `covariates` attributes be used convert the predictions to scale of the original data?
            cv_ix: index of the data value to remove in the case of cross-validation

        Returns:
            dataframe with predicted values and standard deviations at each location
        """
        self.i = i
        pred_cov = self._pred_cov(pcoords)
        pred_cross_cov = self._pred_cross_cov(pcoords, cv_ix=cv_ix)
        joint_cov = self._joint_cov(cv_ix=cv_ix)
        data_values = [
            self.mf.fields[i].values_main.copy() for i in range(self.n_procs)
        ]
        if cv_ix is not None:
            pcoords = pd.DataFrame({"d1": pcoords[0], "d2": pcoords[1]}, index=[0])
            data_values[self.i] = np.delete(data_values[self.i], cv_ix, axis=0)
        else:
            try:
                _verify_model(pred_cov, pred_cross_cov, joint_cov)
            except LinAlgError:
                warnings.warn(
                    "Prediction joint covariance matrix is not positive definte; model"
                    " technically invalid."
                )
        stacked_data = np.hstack(data_values)
        cov_weights = cho_solve(
            cho_factor(joint_cov, lower=True, overwrite_a=True, check_finite=False),
            pred_cross_cov.copy(),
            overwrite_b=True,
            check_finite=False,
        ).T
        pred_var_cov_mat = pred_cov - np.matmul(cov_weights, pred_cross_cov)

        df_pred = pcoords.copy()
        df_pred["pred"] = np.matmul(cov_weights, stacked_data)
        df_pred["pred_err"] = np.nan_to_num(np.sqrt(np.diagonal(pred_var_cov_mat)))

        # NOTE: may choose to return the full variance-covariance matrix in the future
        if postprocess:
            df_pred.rename(columns={"d1": "lat", "d2": "lon"}, inplace=True)
            return self._postprocess_predictions(df_pred)
        else:
            ds = df_pred.set_index(pcoords.columns.values.tolist()).to_xarray()
            try:
                np.isnan(self.mf.fields[self.i].timestamp)
                return ds
            except TypeError:
                return ds.assign_coords(
                    coords={"time": np.datetime64(self.mf.fields[self.i].timestamp)}
                )

    def _pred_cov(self, pcoords: np.ndarray) -> np.ndarray:
        """Compute the variance-covariance matrix for prediction locations for the specified process."""
        dists = distance_matrix(
            pcoords,
            pcoords,
            units=self.dist_units,
            fast_dist=self.fast_dist,
        )
        return self.mod.covariance(self.i, dists, use_nugget=True)

    def _pred_cross_cov(self, pcoords: np.ndarray, cv_ix: int = None) -> np.ndarray:
        """Computes the covariance and cross-covariance vectors between data and prediction locations."""
        dists = [
            distance_matrix(
                f.coords_main, pcoords, units=self.dist_units, fast_dist=self.fast_dist
            )
            for f in self.mf.fields
        ]
        if cv_ix is not None:
            dists[self.i] = np.delete(dists[self.i], cv_ix, axis=0)
        cov_vecs = list()
        for j in range(self.n_procs):
            if self.i == j:
                cov_vecs.append(
                    self.mod.covariance(self.i, dists[self.i], use_nugget=True)
                )
            else:
                cov_vecs.append(self.mod.cross_covariance(self.i, j, dists[j]))
        return np.vstack(cov_vecs)

    def _joint_cov(self, cv_ix: int = None) -> np.ndarray:
        """Compute each block component with each block describing the dependence within a process or between processes; return the block-covariance matrix."""
        blocks = dict()
        for i in range(self.n_procs):
            for j in range(self.n_procs):
                if i <= j:
                    dists = self.mf.calc_dist_matrix(
                        (i, j), self.dist_units, self.fast_dist, main=True
                    )
                    if i == j:
                        blocks[f"{i}{j}"] = self.mod.covariance(i, dists)
                    else:
                        blocks[f"{i}{j}"] = self.mod.cross_covariance(i, j, dists)
                else:
                    # blocks in lower-triangle are the transpose of the upper-triangle
                    blocks[f"{i}{j}"] = blocks[f"{j}{i}"].T.copy()
        if cv_ix is not None:
            for i in range(self.n_procs):
                for j in range(self.n_procs):
                    if i == self.i:
                        blocks[f"{i}{j}"] = np.delete(blocks[f"{i}{j}"], cv_ix, axis=0)
                    if j == self.i:
                        blocks[f"{i}{j}"] = np.delete(blocks[f"{i}{j}"], cv_ix, axis=1)

        return np.block(
            [
                [blocks[f"{i}{j}"] for j in range(self.n_procs)]
                for i in range(self.n_procs)
            ]
        )

    def _postprocess_predictions(self, df: pd.DataFrame) -> xr.Dataset:
        """Convert prediction results to a dataset and transform to original data scale."""
        ds = df.set_index(["lon", "lat"]).to_xarray()
        df_ = df[["lon", "lat"]].copy()

        # Transform predictions and errors to original data scale
        ds *= self.mf.fields[self.i].ds.attrs["scale_fact"]

        # Add back the constant spatial mean used for standardization
        ds["pred"] += self.mf.fields[self.i].ds.attrs["spatial_mean"]

        # Prepare the spatial covariate
        if self.covariates is None:
            covariates = df[["lon", "lat"]].copy()
        else:
            ds["covariates"] = self.covariates.sel(
                time=self.mf.fields[self.i].timestamp
            )
            df_covariates = (
                ds.to_dataframe()
                .reset_index()
                .merge(df_, on=["lon", "lat"], how="right")
                .dropna(subset=["covariates"])
            )
            # realign coordinates
            df_ = df_covariates[["lon", "lat"]].copy()
            covariates = df_covariates[["covariates"]].copy()
        # standardize each covariate using mean and scale from fitting (so covariates are the same at data locations)
        for i, covar in enumerate(covariates):
            covariates[covar] = (
                covariates[covar]
                - self.mf.fields[self.i].ds.attrs["covariate_means"][i]
            ) / self.mf.fields[self.i].ds.attrs["covariate_scales"][i]

        # Add back the spatial trend surface
        df_["ols_mean"] = (
            self.mf.fields[self.i].ds.attrs["spatial_model"].predict(covariates)
        )
        da = (
            df_.set_index(["lon", "lat"])
            .to_xarray()
            .assign_coords(
                coords={"time": np.datetime64(self.mf.fields[self.i].timestamp)}
            )
        )
        ds["pred"] += da["ols_mean"]

        # Add back the temporal trend value
        ds["pred"] += self.mf.fields[self.i].ds.attrs["temporal_trend"]

        return ds

    def cross_validation(
        self,
        i: int,
        postprocess: bool = True,
    ) -> pd.DataFrame:
        """Leave one out cross-validation at each data location; i.e., prediction at each data location with the corresponding data value withheld.

        Parameters: see __call__

        Returns:
            dataframe containg prediction residuals
        """
        if postprocess:
            coord_names = ["lat", "lon"]
        else:
            coord_names = ["d1", "d2"]

        # prepare data the data; each row will be withheld then predicted
        data = pd.DataFrame(
            np.hstack(
                (
                    self.mf.fields[i].coords_main,
                    np.atleast_2d(self.mf.fields[i].values_main).T,
                )
            ),
            columns=coord_names + ["data"],
        )

        # compute predictions row by row
        list_ds = list()
        for ix, p in data[coord_names].iterrows():
            list_ds.append(
                self.__call__(
                    i,
                    p.values,
                    postprocess=postprocess,
                    cv_ix=ix,
                )
            )

        # merge predictions with data and compute difference
        df = (
            xr.merge(list_ds)
            .to_dataframe()
            .reset_index()
            .dropna(subset=["pred"])
            .merge(data, on=coord_names, how="outer")
        )
        df["residual"] = df["data"] - df["pred"]
        col_names = coord_names + ["data", "pred", "residual", "pred_err"]
        return df[col_names]


def _verify_model(
    pred_cov: np.ndarray,
    pred_cross_cov: np.ndarray,
    joint_cov: np.ndarray,
):
    """Check that the overarching joint covariance matrix for a given prediction location is positive definite using the Cholesky decompostion."""
    cho_factor(
        np.vstack(
            [
                np.hstack([pred_cov, pred_cross_cov.T]),
                np.hstack([pred_cross_cov, joint_cov]),
            ]
        ),
        overwrite_a=True,
    )


def prediction_coords(
    extents: tuple = (-125, -65, 22, 58), lon_res: float = 0.5, lat_res: float = 0.5
) -> np.ndarray:
    """Produces prediction coordinates (land only)."""
    grid = GridConfig(extents=extents, lon_res=lon_res, lat_res=lat_res)
    df = land_grid(grid)
    return df.reset_index()[["lat", "lon"]]
