## Classes for cokriging, covariance kernels, etc.
import numpy as np
import xarray
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances


def expand_grid(*args):
    """
    Returns an array of all combinations of elements in the supplied vectors.
    """
    return np.array(np.meshgrid(*args)).T.reshape(-1, len(args))


def distance_matrix(X1, X2=None, units="km"):
    """
    Computes the geodesic distance among all pairs of points given two sets of coordinates.
    Wrapper for scipy.spatial.distance.cdist using geopy.distance.geodesic as a the metric.

    NOTE: points should be formatted in rows as [lat, lon]
    """
    if X2 is None:
        return cdist(X1, X1, lambda s_i, s_j: getattr(geodesic(s_i, s_j), units))
    else:
        return cdist(X1, X2, lambda s_i, s_j: getattr(geodesic(s_i, s_j), units))


def distance_matrix_fast(X1, X2):
    """
    Computes the Haversine (or Great Circle) distance among all pairs of points (in kilometers).
    """
    EARTH_RADIUS = 6371  # radius in kilometers
    X1_r = np.radians(X1)
    X2_r = np.radians(X2)
    return haversine_distances(X1_r, X2_r) * EARTH_RADIUS


class Cokrige:
    """
    Details and references
    """

    def __init__(
        self,
        kernel=None,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
    ):
        pass

    def covariance(self, kernel, params):
        """
        Computes the cokriging covariance matrix.
        """

    def fit(self):
        """
        Fit model parameters via maximum likelihood estimation.
        """
        pass

    def predict(self, return_se=False):
        """
        Cokriging prediction
        """
        pass


class Kernel:
    """
    Gaussian process kernels.
    """

    def __call__(self):
        pass

    def set_params(self):
        pass

    def get_params(self):
        pass


class Matern(Kernel):
    pass

