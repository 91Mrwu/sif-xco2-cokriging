import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from xarray import Dataset

import spatial_tools
from data_utils import get_main_coords


class VarioConfig:
    """Structure to store configuration parameters for an empirical variogram.

    Parameters:
        dist_units: units in which the distance is measured; default is kilometers
        fast_dist: indicates whether to a faster distance calculation with a slight sacrifice in accuracy; default is True
    """

    def __init__(
        self,
        max_dist: float,
        n_bins: int,
        n_procs: int = 2,
        kind: str = "Semivariogram",
        dist_units: str = "km",
        fast_dist: bool = True,
    ) -> None:
        self.max_dist = max_dist
        self.n_bins = n_bins
        self.n_procs = n_procs
        self.kind = kind
        self.dist_units = dist_units
        self.fast_dist = fast_dist
        if self.kind == "Covariogram":
            self.covariogram = True
        else:
            self.covariogram = False


@dataclass
class EmpiricalVariogram:
    """Empirical variogram"""

    df: pd.DataFrame
    config: VarioConfig
    timestamp: str
    timedeltas: list[int]


class Field:
    """
    Stores data values and coordinates for a single process at a fixed time in a data frame.
    """

    def __init__(self, ds: Dataset, covariates: list, timestamp: np.datetime64) -> None:
        self.timestamp = timestamp
        self.data_name, self.var_name = _get_field_names(ds)
        self.ds = _preprocess_ds(ds, timestamp, covariates)
        self.ds_main = get_main_coords(self.ds).sel(time=timestamp)
        df = self.ds.to_dataframe().reset_index().dropna(subset=[self.data_name])
        df_main = (
            self.ds_main.to_dataframe().reset_index().dropna(subset=[self.data_name])
        )
        self.coords = df[["lat", "lon"]].values
        self.coords_main = df_main[["lat", "lon"]].values
        self.values = df[self.data_name].values
        self.values_main = df_main[self.data_name].values
        self.temporal_trend = self.ds.attrs["temporal_trend"]
        self.spatial_trend = df["spatial_trend"].values
        self.spatial_mean = self.ds.attrs["spatial_mean"]
        self.scale_fact = self.ds.attrs["scale_fact"]
        self.variance_estimate = df[self.var_name].values
        self.covariates = df[covariates]
        self.size = len(self.values)

    def to_xarray(self) -> Dataset:
        """Converts the field to an xarray dataset."""
        return (
            pd.DataFrame(
                {
                    "lat": self.coords[:, 0],
                    "lon": self.coords[:, 1],
                    self.data_name: self.values,
                }
            )
            .set_index(["lon", "lat"])
            .to_xarray()
            .assign_coords({"time": np.array(self.timestamp, dtype=np.datetime64)})
        )


class MultiField:
    """
    Stores a multivariate process, each of class Field, along with modelling attributes.

    Parameters:
        datasets: list of xarray datasets
        covariates: list of lists, each containing the names of the covariates for spatial trend removal
        timestamp: the main timestamp for the multifield, timedeltas are with reference to this value
        timedeltas: list of offsets (by month) with elements corresponding to each dataset (negative = back, positive = forward)
    """

    def __init__(
        self,
        datasets: list[Dataset],
        covariates: list[list],
        timestamp: np.datetime64,
        timedeltas: list[int],
    ) -> None:
        _check_length_match(datasets, covariates, timedeltas)
        self.timestamp = np.datetime_as_string(timestamp, unit="D")
        self.timedeltas = timedeltas
        self.datasets = datasets
        self.covariates = covariates
        self.fields = np.array(
            [
                Field(datasets[i], covariates[i], self._apply_timedelta(timedeltas[i]))
                for i in range(len(datasets))
            ]
        )
        self.n_procs = len(self.fields)
        self.n_data = self._count_data()
        # self.joint_data_vec = np.hstack((self.field_1.values, self.field_2.values))
        # self.joint_std_inverse = np.float_power(
        #     np.hstack((self.field_1.temporal_std, self.field_2.temporal_std)), -1
        # )

    def _apply_timedelta(self, timedelta: int) -> str:
        """Returns timestamp with month offset by timedelta as string."""
        t0 = datetime.strptime(self.timestamp, "%Y-%m-%d")
        return (t0 + relativedelta(months=timedelta)).strftime("%Y-%m-%d")

    def _count_data(self) -> int:
        """Returns the total number of data values across all fields."""
        return np.sum([f.size for f in self.fields])

    def calc_dist_matrix(
        self, ids: tuple, units: str, fast_dist: bool, main: bool = False
    ) -> np.ndarray:
        assert len(ids) == 2
        if main:
            coord_list = [self.fields[i].coords_main for i in ids]
        else:
            coord_list = [self.fields[i].coords for i in ids]
        return spatial_tools.distance_matrix(
            *coord_list, units=units, fast_dist=fast_dist
        )

    def _variogram_cloud(self, i: int, j: int, config: VarioConfig) -> pd.DataFrame:
        """Calculate the (cross-) variogram cloud for corresponding field id's."""
        dist = self.calc_dist_matrix((i, j), config.dist_units, config.fast_dist)
        if i == j:
            # marginal-variogram
            idx = np.triu_indices(dist.shape[0], k=1, m=dist.shape[1])
            dist = dist[idx]
            cloud = _cloud_calc(self.fields[[i, i]], config.covariogram)[idx]
        else:
            # cross-variogram
            dist = dist.flatten()
            cloud = _cloud_calc(self.fields[[i, j]], config.covariogram).flatten()

        assert cloud.shape == dist.shape
        return pd.DataFrame({"distance": dist, "variogram": cloud})

    def get_variogram(self, i: int, j: int, config: VarioConfig) -> pd.DataFrame:
        """Compute the (cross-) variogram of the specified kind for the pair of fields (i, j). Return as a dataframe with bin averages and bin counts."""
        df_cloud = self._variogram_cloud(i, j, config)
        # NOTE: if computation becomes slow, max_dist filter could be applied before computing the cloud values
        df_cloud = df_cloud[df_cloud.distance <= config.max_dist]
        bin_centers, bin_edges = _construct_variogram_bins(df_cloud, config.n_bins)
        df_cloud["bin_center"] = pd.cut(
            df_cloud["distance"], bin_edges, labels=bin_centers, include_lowest=True
        )
        df = (
            df_cloud.groupby("bin_center")["variogram"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "bin_mean", "count": "bin_count"})
            .reset_index()
        )
        # convert bins from categories to numeric
        df["bin_center"] = df["bin_center"].astype("string").astype("float")
        if (df["bin_count"] < 30).any():
            warnings.warn(
                f"WARNING: Fewer than 30 pairs used for at least one bin in variogram"
                f" calculation."
            )
        # establish multi-index using field id's
        df["i"], df["j"] = i, j
        return df.set_index(["i", "j", df.index])

    def empirical_variograms(self, config: VarioConfig) -> EmpiricalVariogram:
        """Compute empirical variogram of the specified kind for each field in the multifield and the cross-variogram of the specified kind for each pair of fields.

        Parameters:
            max_dist: maximum distance (in units corresponding to the multifield) across which variograms will be computed
            n_bins: number of bins to use when averaging variogram cloud values
            kind: either `semivariogram` (default) or `covariogram`
        Returns:
            df: multi-index datafame with first two indices corresponding to the field ids used in the calculation
        """
        variograms = [
            self.get_variogram(i, j, config)
            for i in range(self.n_procs)
            for j in range(self.n_procs)
            if i <= j
        ]
        return EmpiricalVariogram(
            pd.concat(variograms), config, self.timestamp, self.timedeltas
        )


def _check_length_match(*args):
    """Check that each input list has the same length."""
    if len({len(i) for i in args}) != 1:
        raise ValueError("Not all lists have the same length")


def _get_field_names(ds: Dataset):
    """Returns data and estimated variance names from dataset."""
    var_name = [name for name in list(ds.keys()) if "_var" in name][0]
    data_name = var_name.replace("_var", "")
    return data_name, var_name


def _median_abs_dev(x: np.ndarray) -> float:
    """Returns the median absolute deviation of the input array.

    Assumes the array is Normally distributed. See https://en.wikipedia.org/wiki/Median_absolute_deviation for details."""
    k = 1.4826  # scale factor assuming a normal distribution
    return k * np.nanmedian(np.abs(x - np.nanmedian(x)))


def _preprocess_ds(ds: Dataset, timestamp: str, covariates: list) -> Dataset:
    """Apply data transformations and compute surface mean and standard deviation."""
    data_name, _ = _get_field_names(ds)
    ds_copy = ds.copy()

    # Remove linear trend over time
    ds_copy["temporal_trend"] = spatial_tools.fit_linear_trend(ds_copy[data_name])
    ds_copy[data_name] = ds_copy[data_name] - ds_copy["temporal_trend"]

    # Select data at timestamp (fields are spatial only)
    ds_field = ds_copy.sel(time=timestamp)
    ds_field.attrs["temporal_trend"] = ds_field["temporal_trend"].values

    # Remove the spatial trend by OLS
    ds_field["spatial_trend"] = spatial_tools.fit_ols(ds_field, data_name, covariates)
    ds_field[data_name] = ds_field[data_name] - ds_field["spatial_trend"]

    # Standardize the residuals
    ds_field.attrs["spatial_mean"] = np.nanmean(ds_field[data_name].values)
    ds_field.attrs["scale_fact"] = np.nanstd(ds_field[data_name].values)
    # ds_field.attrs["scale_fact"] = _median_abs_dev(ds_field[data_name].values)
    ds_field[data_name] = (
        ds_field[data_name] - ds_field.attrs["spatial_mean"]
    ) / ds_field.attrs["scale_fact"]

    return ds_field


def _cloud_calc(fields: list[Field], covariogram: bool) -> np.ndarray:
    """Calculate the semivariogram or covariogram for all point pairs."""
    center = lambda f: f.values - f.values.mean()
    residuals = [center(f) for f in fields]
    if covariogram:
        cloud = np.multiply.outer(*residuals)
    else:
        cloud = 0.5 * (np.subtract.outer(*residuals)) ** 2
    return cloud


def _construct_variogram_bins(
    df_cloud: pd.DataFrame, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Paritions the domain of a variogram cloud into `n_bins` bins; first bin extended to zero."""
    # use min non-zero dist for consistincy between variograms and cross-variograms
    min_dist = df_cloud[df_cloud["distance"] > 0]["distance"].min()
    max_dist = df_cloud["distance"].max()
    bin_centers = np.linspace(min_dist, max_dist, n_bins)
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = np.arange(min_dist - 0.5 * bin_width, max_dist + bin_width, bin_width)
    # check that bin centers are actually centered
    if not np.allclose((bin_edges[1:] + bin_edges[:-1]) / 2, bin_centers):
        warnings.warn("WARNING: variogram bins are not centered.")
    bin_edges[0] = 0
    return bin_centers, bin_edges
