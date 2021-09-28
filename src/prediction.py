"""
Steps:
[x] recover the data values along the main grid points only (i.e., original 4x5-degree grid)
[x] setup prediction locations with resultion 0.5x0.5-degree (land only)
[x] precompute the grand-covariance matrix blocks for all data locations
[x] for a given prediction location, find the data indices within a max_dist for each dataset
    3.1 if there are more than N points in that window, then subsample [rather than subsample, shrink the window]
[x] collect the block covariance matrix and the stacked data vector at the correct indices
[x] compute covariance arrays between prediction locations and data locations
[x] verify local model using cholesky decomp (check that joint cov mat is not singular)
[x] compute prediction of residual process
[x] compute the uncertainty for the residuals
[x] paralleize across prediction grid
[] post-process the predicted residuals
    [] predict the mean surface at prediction (standardized) locations or (standardized) evi [need to update EVI dataset for this]
    [] multiply by the scale factor and add the (constant) spatial mean
    [] add the spatial trend surface
    [] add the temporal trend value
    [] multiply diagonal elements of uncertainty value by scale factor squared, then take the square root of that value (gives pred uncertainty)

TODO: 
 - allow for prediction of process [i] rather than process 0 only
"""

import warnings
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from fields import MultiField
from model import FullBivariateMatern
from spatial_tools import distance_matrix
from data_utils import GridConfig, land_grid

# number of paritions for parallel computation
NCORES = cpu_count()


class Predictor:
    """Multivariate prediction framework."""

    def __init__(
        self,
        mod: FullBivariateMatern,
        mf: MultiField,
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
        self.dist_units = dist_units
        self.fast_dist = fast_dist
        self.Sigma = self._cov_blocks()

    def __call__(
        self, pcoords: np.ndarray, max_dist: float = 1e3, partitions: int = None
    ) -> pd.DataFrame:
        """Apply the multivariate local prediction at each location in the specified prediction coordinates.

        Parameters:
            pcoords: prediction coordinates with format [[lat, lon]]
            max_dist: maximum distance at which data values will be included in prediction
            partitions: number of partitions to split prediction locations into for parallelization

        Returns:
            dataframe with predicted values and standard deviations at each location
        """
        c0 = self.mod.covariance(0, 0, use_nugget=False)[0]
        df = pd.DataFrame(pcoords)
        df.columns = ["Latitude", "Longitude"]

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
            df_pred = self._predict_chunk(df, c0, max_dist)

        return df_pred

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
        cov_vecs = [
            self.mod.cross_covariance(0, j, dists[j]) for j in range(1, self.n_procs)
        ]
        cov_vecs.insert(0, self.mod.covariance(0, dists[0], use_nugget=False))
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
        # format the data as a vector, the covariance blocks as a matrix, and the prediction covariance as a vector
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
            )
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
        pred = np.matmul(
            local_pred_cov,
            cho_solve(
                cho_factor(local_cov, lower=True, overwrite_a=True, check_finite=False),
                local_data,
                overwrite_b=True,
                check_finite=False,
            ),
        )
        pred_std = np.sqrt(
            c0
            - np.matmul(
                local_pred_cov,
                cho_solve(
                    cho_factor(
                        local_cov, lower=True, overwrite_a=True, check_finite=False
                    ),
                    local_pred_cov,
                    overwrite_b=True,
                    check_finite=False,
                ),
            )
        )
        return pred, pred_std

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


def prediction_coords(
    extents: tuple = (-125, -65, 22, 58), lon_res: float = 0.5, lat_res: float = 0.5
) -> np.ndarray:
    """Produces prediction coordinates (land only)."""
    grid = GridConfig(extents=extents, lon_res=lon_res, lat_res=lat_res)
    df = land_grid(grid)
    return df.reset_index()[["lat", "lon"]].values
