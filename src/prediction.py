"""
Steps:
[x] recover the data values along the main grid points only (i.e., original 4x5-degree grid)
[x] setup prediction locations with resultion 0.5x0.5-degree (land only)
[x] precompute the grand-covariance matrix blocks for all data locations
[x] for a given prediction location, find the data indices within a max_dist for each dataset
    [ ] if there are more than N points in that window, then subsample [rather than subsample, shrink the window]
[x] collect the block covariance matrix and the stacked data vector at the correct indices
[x] compute covariance arrays between prediction locations and data locations
[x] verify local model using cholesky decomp (check that joint cov mat is not singular)
[x] compute prediction of residual process
[x] compute the uncertainty for the residuals
[x] paralleize across prediction grid
[x] post-process the predicted residuals
    [x] predict the mean surface at prediction (standardized) locations or (standardized) evi [need to update EVI dataset for this]
    [x] multiply predictions and errors by the scale factor
    [x] add the (constant) spatial mean to predictions
    [x] add the spatial trend surface to predictions
    [x] add the temporal trend value to predictions

TODO: 
 [x] allow for prediction of process [i] rather than process 0 only
 [x] prediction results plotting function
"""

import warnings
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from xarray import Dataset, DataArray
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from fields import MultiField
from model import MultivariateMatern
from spatial_tools import distance_matrix
from stat_tools import standardize
from data_utils import GridConfig, land_grid

# number of paritions for parallel computation
NCORES = cpu_count()


class Predictor:
    """Multivariate prediction framework."""

    def __init__(
        self,
        mod: MultivariateMatern,
        mf: MultiField,
        covariates: DataArray = None,
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
        self.Sigma = self._cov_blocks()

    def __call__(
        self,
        i: int,
        pcoords: np.ndarray,
        max_dist: float = 1e3,
        partitions: int = None,
        postprocess: bool = True,
    ) -> pd.DataFrame:
        """Apply the multivariate local prediction at each location in the specified prediction coordinates.

        Parameters:
            i: index of the process to be predicted
            pcoords: prediction coordinates with format [[lat, lon]]
            max_dist: maximum distance at which data values will be included in prediction
            partitions: number of partitions to split prediction locations into for parallelization

        Returns:
            dataframe with predicted values and standard deviations at each location
        """
        self.i = i
        c0 = self.mod.covariance(self.i, 0, use_nugget=True)[0]
        df = pd.DataFrame(pcoords)
        df.columns = ["lat", "lon"]

        if partitions is not None:
            # run prediction in parallel across `n_partitions`
            data_split = list(
                zip(
                    np.array_split(df, partitions),
                    [c0] * partitions,
                    [max_dist] * partitions,
                )
            )
            pool = Pool(NCORES)
            df_pred = pd.concat(pool.starmap(self._predict_chunk, data_split))
            pool.close()
            pool.join()
        else:
            # process full data as a single chunk
            df_pred = self._predict_chunk(df, c0, max_dist)

        if postprocess:
            return self._postprocess_predictions(df_pred)
        else:
            return (
                df_pred.set_index(["lon", "lat"])
                .to_xarray()
                .assign_coords(
                    coords={"time": np.datetime64(self.mf.fields[self.i].timestamp)}
                )
            )

    def _cov_blocks(self) -> dict:
        """Precomputes each block in the block-covariance matrix, with each block describing the dependence within a process or between processes."""
        blocks = dict()
        for i in range(self.n_procs):
            for j in range(self.n_procs):
                if i <= j:
                    h = self.mf.calc_dist_matrix(
                        (i, j), self.dist_units, self.fast_dist, main=True
                    )
                    if i == j:
                        blocks[f"{i}{j}"] = self.mod.covariance(i, h)
                        # add measurement error variance along diagonals
                        # np.fill_diagonal(blocks[i, j], blocks[i, j].diagonal() + sigep[i])
                    else:
                        blocks[f"{i}{j}"] = self.mod.cross_covariance(i, j, h)
        # NOTE: is normalization by data std necessary?
        return blocks

    def _pred_cov(self, dists: list) -> list:
        """Computes the covariance and cross-covariance vectors for a set of local distances about a prediction location."""
        cov_vecs = list()
        for j in range(self.n_procs):
            if self.i == j:
                cov_vecs.append(
                    self.mod.covariance(self.i, dists[self.i], use_nugget=True)
                )
            else:
                cov_vecs.append(self.mod.cross_covariance(self.i, j, dists[j]))
        # NOTE: do these values need to be rescaled using the data std?
        return cov_vecs

    def _local_dist_ix(self, s0: np.ndarray, max_dist: float) -> tuple[list, list]:
        """Determines the distances and corresponding indices of all data locations within `max_dist` of the specified prediction location `s0` for each dataset."""
        dists = [
            distance_matrix(
                s0,
                f.coords_main,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            )
            for f in self.mf.fields
        ]
        # NOTE: may need to adjust max_dist if the size of any of these elements is larger than some n_max
        ix = [(d <= max_dist).squeeze() for d in dists]
        local_dists = [d[d <= max_dist] for d in dists]
        for d in local_dists:
            assert d.max() <= max_dist
        return ix, local_dists

    def _local_values(
        self, s0: np.ndarray, max_dist: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For a given prediction location s0, return the data vector and covariance matrix based on locations within `max_dist`."""
        ix, local_dists = self._local_dist_ix(s0, max_dist)
        local_data = [self.mf.fields[i].values_main[ix[i]] for i in range(self.n_procs)]
        local_cov_blocks = dict()
        for i in range(self.n_procs):
            for j in range(self.n_procs):
                if i <= j:
                    local_cov_blocks[f"{i}{j}"] = self.Sigma[f"{i}{j}"][
                        np.ix_(ix[i], ix[j])
                    ]
                else:
                    # blocks in lower-triangle are the transpose of the upper-triangle
                    local_cov_blocks[f"{i}{j}"] = self.Sigma[f"{j}{i}"][
                        np.ix_(ix[j], ix[i])
                    ].T
        # format the data and prediction covarianceas as vectors, and the covariance blocks as a matrix
        local_data = np.hstack(local_data)
        local_pred_cov = np.hstack(self._pred_cov(local_dists))
        local_cov = np.block(
            [
                [local_cov_blocks[f"{i}{j}"] for j in range(self.n_procs)]
                for i in range(self.n_procs)
            ]
        )
        assert local_data.shape[0] == local_cov.shape[0]
        return local_pred_cov, local_cov, local_data

    def _verify_model(
        self,
        c0: float,
        local_pred_cov: np.ndarray,
        local_cov: np.ndarray,
    ):
        """Check that the overarching joint covariance matrix for a given prediction location is positive definite using the Cholesky decompostion."""
        cho_factor(
            np.vstack(
                [
                    np.hstack([c0, local_pred_cov]),
                    np.column_stack([local_pred_cov, local_cov]),
                ]
            ),
            overwrite_a=True,
        )

    @staticmethod
    def _pred_calc(
        c0: float,
        local_pred_cov: np.ndarray,
        local_cov: np.ndarray,
        local_data: np.ndarray,
    ) -> tuple[float, float]:
        """Local prediction and uncertainty calculations."""
        # NOTE: should be safe to use overwrites with no finite check since this will be done in model verification
        cov_weights = cho_solve(
            cho_factor(local_cov, lower=True, overwrite_a=True, check_finite=False),
            local_pred_cov.copy(),
            overwrite_b=True,
            check_finite=False,
        )
        pred = np.matmul(cov_weights, local_data)
        pred_std = np.sqrt(c0 - np.matmul(cov_weights, local_pred_cov))
        return pred, np.nanmax([pred_std, 0.0])

    def _local_prediction(
        self, s0: np.ndarray, c0: float, max_dist: float
    ) -> tuple[float, float]:
        """Compute the predicted value and prediction standard deviation at the specified location."""
        local_pred_cov, local_cov, local_data = self._local_values(s0, max_dist)
        try:
            self._verify_model(c0, local_pred_cov, local_cov)
        except LinAlgError:
            warnings.warn(f"Invalid model at prediction location {s0}. Returning NaN.")
            return np.nan, np.nan
        # NOTE: do we need to scale pred_std by the data std (i.e., pre- post- diag multiply)?
        return self._pred_calc(c0, local_pred_cov, local_cov, local_data)

    def _predict_chunk(self, df_chunk: pd.DataFrame, c0: float, max_dist: float):
        df_chunk[["pred", "pred_err"]] = df_chunk.apply(
            lambda s0: self._local_prediction(s0.values, c0, max_dist),
            axis=1,
            result_type="expand",
        )
        return df_chunk

    def _postprocess_predictions(self, df: pd.DataFrame) -> Dataset:
        """Convert prediction results to a dataset and transform to original data scale."""
        ds = df.set_index(["lon", "lat"]).to_xarray()
        df_ = df[["lon", "lat"]].copy()

        # Transform predictions and errors to original data scale
        ds = ds * self.mf.fields[self.i].ds.attrs["scale_fact"]

        # Add back the constant spatial mean used for standardization
        ds["pred"] += self.mf.fields[self.i].ds.attrs["spatial_mean"]

        # Prepare the spatial covariate
        if self.covariates is None:
            covariates = df[["lon", "lat"]]
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
            df_ = df_covariates[["lon", "lat"]].copy()
            covariates = df_covariates[["covariates"]]
        # standardize each covariate (each column)
        covariates = covariates.apply(lambda x: standardize(x), axis=0)

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


def prediction_coords(
    extents: tuple = (-125, -65, 22, 58), lon_res: float = 0.5, lat_res: float = 0.5
) -> np.ndarray:
    """Produces prediction coordinates (land only)."""
    grid = GridConfig(extents=extents, lon_res=lon_res, lat_res=lat_res)
    df = land_grid(grid)
    return df.reset_index()[["lat", "lon"]].values
