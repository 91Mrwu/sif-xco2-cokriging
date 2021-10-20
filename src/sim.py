import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky

from model import MultivariateMatern
from fields import MultiField


class CartesianGrid:
    """Regular Cartesian grid in Euclidean space."""

    def __init__(
        self, xbounds: tuple = (0, 1), ybounds: tuple = (0, 1), xcount=51, ycount=51
    ) -> None:
        xcoords = np.linspace(*xbounds, num=xcount)
        ycoords = np.linspace(*ybounds, num=ycount)
        self.coords = pd.DataFrame(
            self._expand_grid(xcoords, ycoords), columns=["x", "y"]
        )
        self.count = len(self.coords)
        self.dist = cdist(self.coords.values, self.coords.values)

    def _expand_grid(self, *args) -> np.ndarray:
        """Returns an array of all combinations of elements in the supplied vectors."""
        return np.array(np.meshgrid(*args)).T.reshape(-1, len(args))


class BivariateRandomField:
    """Simulate and sample a bivariate Gaussian random field from the supplied model."""

    def __init__(
        self, model: MultivariateMatern, grid: CartesianGrid, seed: int = None
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mod = model
        self.grid = grid
        self.coords = grid.coords
        self.cmat = self._joint_cov_matrix()
        self.chol_fact_lower = cholesky(self.cmat, lower=True)
        self.fields = self._simulate()

    def _joint_cov_matrix(self) -> dict:
        """Precomputes each block in the block-covariance matrix, with each block describing the dependence within a process or between processes."""
        c11 = self.mod.covariance(0, self.grid.dist)
        c22 = self.mod.covariance(1, self.grid.dist)
        c12 = self.mod.cross_covariance(0, 1, self.grid.dist)
        return np.block([[c11, c12], [c12.T, c22]])

    def _simulate(self) -> list:
        noise_vec = self.rng.standard_normal(2 * self.grid.count)
        sim_data = np.atleast_2d(self.chol_fact_lower @ noise_vec)
        field_split = [
            sim_data[:, : self.grid.count].T,
            sim_data[:, self.grid.count :].T,
        ]
        return [
            pd.DataFrame(
                np.hstack((self.coords.values, field_split[i])),
                columns=["x", "y", "value"],
            )
            for i in range(2)
        ]

    def _split_samp_coords(self, size: int, seed: int) -> list:
        """Sample locations where half the samples are co-located and half are not."""
        samp_size_extended = int(np.floor(1.5 * size))
        samp_size_coloc = int(np.ceil(size / 2))
        samp_size_mismatch = size - samp_size_coloc
        assert samp_size_extended >= samp_size_coloc + 2 * samp_size_mismatch

        coords = self.coords.sample(
            n=samp_size_extended, random_state=seed, replace=False
        )
        co_coords = coords.iloc[:samp_size_coloc, :]
        mis_coords = [
            coords.iloc[samp_size_coloc : samp_size_coloc + samp_size_mismatch, :],
            coords.iloc[samp_size_coloc + samp_size_mismatch :, :],
        ]
        return [pd.concat((co_coords, mis_coords[i])) for i in range(2)]

    def sample(
        self,
        size: int = None,
        frac: float = None,
        epsilon: list = [0],
        seed: int = None,
    ) -> list:
        """
        Parameters:
            size: size of the required sample
            frac: fraction of the total simulated data to sample (overwrites size)
            epsilon: measurement error scale (std. dev.) for each data process
            seed: sampling seed can be different from simulation seed, but is the same by default
        """
        if frac is not None:
            size = int(np.ceil(frac * self.grid.count))
        assert (
            1.5 * size <= self.grid.count
        ), "Sample size is too large for semi-colocated sampling scheme."
        epsilon = np.array(epsilon)
        if epsilon.size == 1:
            epsilon = np.repeat(epsilon, 2)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            seed = self.seed

        coords = self._split_samp_coords(size, seed)
        samples = [pd.merge(self.fields[i], coords[i]) for i in range(2)]
        # apply measurement error
        for i, df in enumerate(samples):
            df["value"] += self.rng.normal(scale=epsilon[i], size=size)
            df.rename(columns={"value": f"Z{i}"}, inplace=True)
        return samples

    def to_xarray(self, samples: list = None) -> xr.Dataset:
        if samples is None:
            for i, df in enumerate(self.fields):
                df.rename(columns={"value": f"Y{i}"}, inplace=True)
            return pd.merge(*self.fields, how="outer").set_index(["x", "y"]).to_xarray()
        else:
            return pd.merge(*samples, how="outer").set_index(["x", "y"]).to_xarray()

    def to_fields(self, samples: list, i: int = None) -> MultiField:
        """Format bivariate samples as a MultiField."""
        ds = self.to_xarray(samples)
        datasets = list()
        for j in range(2):
            ds[f"Z{j}_var"] = np.nan
            datasets.append(ds[[f"Z{j}", f"Z{j}_var"]])
        if i is None:
            return MultiField(datasets, None, np.nan, None, type="sim")
        else:
            return MultiField([datasets[i]], None, np.nan, None, type="sim")
