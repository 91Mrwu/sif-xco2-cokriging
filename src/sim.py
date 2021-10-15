"""
Create a class that simulates a bivariate Gaussian random field using the multivariate Matern model.
"""

import numpy as np
from pandas import DataFrame
from xarray import Dataset
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

from model import MultivariateMatern


# Establish a spatial grid
class CartesianGrid:
    def __init__(
        self, xbounds: tuple = (0, 1), ybounds: tuple = (0, 1), xcount=100, ycount=100
    ) -> None:
        xcoords = np.linspace(*xbounds, num=xcount)
        ycoords = np.linspace(*ybounds, num=ycount)
        self.coords = self.expand_grid(xcoords, ycoords)
        self.dist = cdist(self.coords, self.coords)

    def expand_grid(self, *args) -> np.ndarray:
        """Returns an array of all combinations of elements in the supplied vectors."""
        return np.array(np.meshgrid(*args)).T.reshape(-1, len(args))


class Sim:
    def __init__(self, model: MultivariateMatern, grid: CartesianGrid) -> None:
        self.mod = model
        self.grid = grid
        self.cmat = self._joint_cov_matrix()

    def _joint_cov_matrix(self) -> dict:
        """Precomputes each block in the block-covariance matrix, with each block describing the dependence within a process or between processes."""
        c11 = self.mod.covariance(0, self.grid.dist)
        c22 = self.mod.covariance(1, self.grid.dist)
        c12 = self.mod.cross_covariance(0, 1, self.grid.dist)
        return np.block([[c11, c12], [c12.T, c22]])

    def simulate(self):
        pass


# Calculate the Cholesky factor

# Multiply the Chol. factor by a standard Gaussian noise vector aligned with data locations

# Create a simulation class that stores data as a dataframe with option to convert to dataset
