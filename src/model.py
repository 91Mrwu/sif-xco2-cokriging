import warnings

# TODO: implement numba for speed
from numba import njit

# from numba.experimental import jitclass
# from numba import int32, float32
# import numba_scipy

import numpy as np
import pandas as pd
import scipy.special as sps
from scipy.optimize import minimize

from spatial_tools import pre_post_diag, get_group_ids
from fields import EmpiricalVariogram


class MarginalParam:
    """Multivariate marginal Matern covariance parameter."""

    def __init__(
        self, name: str, default: float, bounds: tuple, n_procs: int = 2
    ) -> None:
        self.name = name
        self.n_procs = n_procs
        self.default = default
        self.bounds = bounds
        self.values = np.nan * np.zeros((n_procs, n_procs))
        np.fill_diagonal(self.values, default)

    def get_names(self):
        return [f"{self.name}_{i+1}{i+1}" for i in range(self.n_procs)]

    def get_values(self):
        return self.values.diagonal()

    def set_values(self, x: np.ndarray):
        np.fill_diagonal(self.values, x)
        return self

    def reset_values(self):
        np.fill_diagonal(self.values, self.default)
        return self

    def count_params(self):
        return len(self.get_values())

    def to_dataframe(self):
        df = (
            pd.DataFrame.from_dict(
                dict(zip(self.get_names(), self.get_values())),
                orient="index",
                columns=["value"],
            )
            .reset_index()
            .rename(columns={"index": "name"})
        )
        df["bounds"] = [self.bounds] * len(df)
        return df


class CrossParam(MarginalParam):
    """Multivariate Matern covariance parameter with cross dependence."""

    def __init__(
        self, name: str, default: float, bounds: tuple, n_procs: int = 2
    ) -> None:
        super().__init__(name, default, bounds, n_procs=n_procs)
        self._triu_index = np.triu_indices(n_procs)
        self.values[self._triu_index] = default

    def get_names(self):
        return [
            f"{self.name}_{i+1}{j+1}"
            for i in range(self.n_procs)
            for j in range(self.n_procs)
            if i <= j
        ]

    def get_values(self):
        return self.values[self._triu_index]

    def set_values(self, x: np.ndarray):
        self.values[self._triu_index] = x
        return self

    def reset_values(self):
        self.values[self._triu_index] = self.default
        return self


class RhoParam(CrossParam):
    """Multivariate Matern covariance parameter with cross dependence only."""

    def __init__(
        self, name: str, default: float, bounds: tuple, n_procs: int = 2
    ) -> None:
        super().__init__(name, default, bounds, n_procs=n_procs)
        self._triu_index = np.triu_indices(n_procs, k=1)
        self.values[self._triu_index] = default

    def get_names(self):
        return [
            f"{self.name}_{i+1}{j+1}"
            for i in range(self.n_procs)
            for j in range(self.n_procs)
            if i < j
        ]


class MaternParams:
    """Multivariate Matern covariance parameters.

    Formulation:
        sigma: process specific standard deviation
        nu: process specific smoothess
        len_scale: process specific length scale
        nugget: process specific squared nugget (i.e., tau^2)
        rho: co-located cross-correlation coefficient(s)
    """

    def __init__(self, n_procs: int = 2) -> None:
        self.n_procs = n_procs
        self.sigma = MarginalParam("sigma", 1.0, (0.4, 3.5), n_procs=n_procs)
        self.nu = CrossParam("nu", 1.5, (0.2, 3.5), n_procs=n_procs)
        self.len_scale = CrossParam("len_scale", 5e2, (1e2, 2e3), n_procs=n_procs)
        self.nugget = MarginalParam("nugget", 0.0, (0.0, 0.2), n_procs=n_procs)
        self.rho = RhoParam("rho", 0.0, (-1.0, 1.0), n_procs=n_procs)
        self._params = [self.sigma, self.nu, self.len_scale, self.nugget, self.rho]
        self.n_params = 0
        for p in self._params:
            self.n_params += p.count_params()

    def to_dataframe(self):
        df_list = [p.to_dataframe() for p in self._params]
        return pd.concat(df_list, ignore_index=True)

    def get_values(self):
        return self.to_dataframe()["value"].values

    def set_values(self, x: np.ndarray):
        if len(x) != self.n_params:
            raise ValueError("Incorrect number of parameters in input array.")
        for p in self._params:
            n_params = p.count_params()
            vals, x = x[:n_params], x[n_params:]
            p.set_values(vals)
        return self

    def reset_values(self):
        for p in self._params:
            p.reset_values()
        return self

    def get_bounds(self):
        return self.to_dataframe()["bounds"].values

    def set_bounds(self, **kwargs):
        for name, bounds in kwargs.items():
            try:
                param = getattr(self, name)
                param.bounds = bounds
            except AttributeError:
                raise AttributeError(f"`{name}` is not a valid parameter.")
        return self


# TODO: add method to check parameter validity (overall pos def and paper specs)
class FullBivariateMatern:
    """Full bivariate Matern covariance model (Gneiting et al., 2010).

    Notes:
    - Parameterization follows Rassmussen and Williams (2006; see MaternParams)
    """

    def __init__(self) -> None:
        self.n_procs = 2
        self.params = MaternParams(n_procs=2)

    def correlation(self, i: int, j: int, h: np.ndarray) -> np.ndarray:
        r"""Matern correlation function.

        Parameters:
            i, j: process indices
            h: array of spatial separation distances (lags), e.g., distance matrix

        .. math::
            \rho(h) =
            \frac{2^{1-\nu}}{\Gamma\left(\nu\right)} \cdot
            \left(\sqrt{2\nu}\cdot\frac{h}{\ell}\right)^{\nu} \cdot
            \mathrm{K}_{\nu}\left(\sqrt{2\nu}\cdot\frac{h}{\ell}\right)
        """
        nu = self.params.nu.values[i, j]
        len_scale = self.params.len_scale.values[i, j]

        # TODO: add check so that negative distances and correlation values yeild warning.
        h = np.array(np.abs(h), dtype=np.double)
        # calculate by log-transformation to prevent numerical errors
        h_positive_scaled = h[h > 0.0] / len_scale
        corr = np.ones_like(h)
        corr[h > 0.0] = np.exp(
            (1.0 - nu) * np.log(2)
            - sps.gammaln(nu)
            + nu * np.log(np.sqrt(2.0 * nu) * h_positive_scaled)
        ) * sps.kv(nu, np.sqrt(2.0 * nu) * h_positive_scaled)
        # if nu >> 1 we get errors for the farfield, there 0 is approached
        corr[np.logical_not(np.isfinite(corr))] = 0.0
        # Matern correlation is positive
        corr = np.maximum(corr, 0.0)
        return corr

    def covariance(self, i: int, h: np.ndarray) -> np.ndarray:
        cov = self.params.sigma.values[i, i] ** 2 * self.correlation(i, i, h)
        cov[h == 0] += self.params.nugget.values[i, i]
        return cov

    def cross_covariance(self, i: int, j: int, h: np.ndarray) -> np.ndarray:
        return (
            self.params.rho.values[i, j]
            * np.nanprod(self.params.sigma.values)
            * self.correlation(i, j, h)
        )

    def semivariance(self, i: int, h: np.ndarray) -> np.ndarray:
        return (
            self.params.sigma.values[i, i] ** 2 * (1.0 - self.correlation(i, i, h))
            + self.params.nugget.values[i, i]
        )

    def cross_semivariance(self, i: int, j: int, h: np.ndarray) -> np.ndarray:
        sill = 0.5 * np.nansum(
            self.params.sigma.values ** 2 + self.params.nugget.values
        )
        return sill - self.cross_covariance(i, j, h)

    def get_variogram(self, i: int, j: int, h: np.ndarray, kind: str) -> pd.DataFrame:
        """Document!"""
        if kind == "covariogram":
            if i == j:
                v = self.covariance(i, h)
            else:
                v = self.cross_covariance(i, j, h)
        else:
            if i == j:
                v = self.semivariance(i, h)
            else:
                v = self.cross_semivariance(i, j, h)
        df = pd.DataFrame({"distance": h, "variogram": v, "i": i, "j": j})
        return df.set_index(["i", "j", df.index])

    def variograms(self, h: np.ndarray, kind: str = "semivariogram") -> pd.DataFrame:
        """Produce modelled variograms and cross-variogram(s) of the specified kind for the given separation distances."""
        variograms = [
            self.get_variogram(i, j, h, kind)
            for i in range(self.n_procs)
            for j in range(self.n_procs)
            if i <= j
        ]
        return pd.concat(variograms)

    # Model fitting

    @staticmethod
    def _weighted_least_squares(
        ydata: np.ndarray, yfit: np.ndarray, bin_counts: np.ndarray
    ) -> float:
        """Computes the weighted least squares cost specified by Cressie (1985)."""
        zeros = np.argwhere(yfit == 0.0)
        non_zero = np.argwhere(yfit != 0.0)
        wls = np.zeros_like(yfit)
        wls[zeros] = (
            bin_counts[zeros] * ydata[zeros] ** 2
        )  # NOTE: is this wrong to do? (for variograms with nonzero nugget, it won't even come up)
        wls[non_zero] = (
            bin_counts[non_zero]
            * ((ydata[non_zero] - yfit[non_zero]) / yfit[non_zero]) ** 2
        )
        return np.sum(wls)

    def _map_fit(self, df_group: pd.DataFrame) -> pd.DataFrame:
        """Creates a new `fit` column with the semivariogram model evaluated at `bin_centers`."""
        i, j = get_group_ids(df_group)
        if i == j:
            df_group["fit"] = self.semivariance(i, df_group["bin_center"].values)
        else:
            df_group["fit"] = self.cross_semivariance(
                i, j, df_group["bin_center"].values
            )
        return df_group

    def _composite_wls(self, p, df_vario: pd.DataFrame) -> float:
        """Composite WLS cost function."""
        self.params.set_values(p)
        df_vario = df_vario.groupby(level=[0, 1]).apply(self._map_fit)
        ydata, yfit, counts = df_vario[["bin_mean", "fit", "bin_count"]].T.values
        non_zero = np.argwhere(yfit != 0.0)
        return _wls(ydata[non_zero], yfit[non_zero], counts[non_zero])

    def fit(self, estimate: EmpiricalVariogram):
        """Fit the model paramters to empirical (cross-) semivariograms *simultaneously* using composite weighted least squares.

        Reference: Extension of Cressie (1985)
        """
        if estimate.config.n_procs != self.n_procs:
            raise ValueError(
                "Number of theoretical processes different from empirical processes."
            )
        init_params = self.params.reset_values().get_values()
        bounds = self.params.get_bounds()
        optim_result = minimize(
            self._composite_wls,
            init_params,
            args=(estimate.df),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if optim_result.success == False:
            warnings.warn("ERROR: optimization did not converge.")
        # TODO: check model validity
        self.params.set_values(optim_result.x)
        self.fit_result = FittedVariogram(self, estimate, optim_result.fun)
        return self

    # Prediction

    def cross_covariance_pred(self, dist_blocks):
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

    def covariance_matrix_pred(self, dist_blocks):
        """Constructs the bivariate Matern covariance matrix.

        NOTE: ask about when to add nugget and error variance along diag
        """
        C_11 = self.kernel_1.sigma ** 2 * self.kernel_1.correlation(
            dist_blocks["block_11"]
        )
        C_12 = (
            self.rho
            * self.kernel_1.sigma
            * self.kernel_2.sigma
            * self.kernel_b.correlation(dist_blocks["block_12"])
        )
        C_21 = (
            self.rho
            * self.kernel_1.sigma
            * self.kernel_2.sigma
            * self.kernel_b.correlation(dist_blocks["block_21"])
        )
        C_22 = self.kernel_2.sigma ** 2 * self.kernel_2.correlation(
            dist_blocks["block_22"]
        )

        # add nugget and measurement error variance along diagonals
        np.fill_diagonal(C_11, C_11.diagonal() + self.kernel_1.nugget + self.sigep_11)
        np.fill_diagonal(C_22, C_22.diagonal() + self.kernel_2.nugget + self.sigep_22)

        # stack blocks into joint covariance matrix and normalize by standard deviation
        cov_mat = np.block([[C_11, C_12], [C_21, C_22]])
        return pre_post_diag(self.fields.joint_std_inverse, cov_mat)


class FittedVariogram:
    """Model parameters and theoretical variogram for the correponding emprical variogram."""

    def __init__(
        self, model: FullBivariateMatern, estimate: EmpiricalVariogram, cost: float
    ) -> None:
        self.config = estimate.config
        self.timestamp = estimate.timestamp
        self.timedeltas = estimate.timedeltas
        self.df_empirical = estimate.df
        h = np.linspace(0, self.df_empirical["bin_center"].max(), 100)
        self.df_theoretical = model.variograms(h)
        self.params = model.params
        self.cost = cost


@njit
def _wls(ydata: np.ndarray, yfit: np.ndarray, bin_counts: np.ndarray) -> float:
    """Computes the weighted least squares cost specified by Cressie (1985)."""
    return np.sum(bin_counts * ((ydata - yfit) / yfit) ** 2)


def _check_cauchyshwarz(covariograms, names):
    """Check the Cauchy-Shwarz inequality."""
    name1 = names[0]
    name2 = names[1]
    cross_name = f"{name1}:{name2}"

    # NOTE: Not exactly C-S if minimum lag is not 0, but should be close
    cov1_0 = covariograms[name1][
        covariograms[name1]["lag"] == np.min(covariograms[name1]["lag"])
    ][name1][0]
    cov2_0 = covariograms[name2][
        covariograms[name2]["lag"] == np.min(covariograms[name2]["lag"])
    ][name2][0]
    cross_cov = covariograms[cross_name][cross_name] ** 2

    if np.any(cross_cov > cov1_0 * cov2_0):
        warnings.warn("WARNING: Cauchy-Shwarz inequality not upheld.")
