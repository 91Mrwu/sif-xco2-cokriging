import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from xarray import Dataset, DataArray
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from sklearn.metrics.pairwise import haversine_distances
from sklearn.linear_model import LinearRegression

from data_utils import get_main_coords
from stat_tools import simple_linear_regression

EARTH_RADIUS = 6371  # radius in kilometers


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

    def __init__(
        self,
        ds: Dataset,
        covariates: list,
        timestamp: np.datetime64,
        type: str = "real",
    ) -> None:
        self.timestamp = timestamp
        self.data_name, self.var_name = _get_field_names(ds)
        if type == "real":
            self.ds = _preprocess_ds(ds, timestamp, covariates)
            self.ds_main = get_main_coords(self.ds).sel(time=timestamp)
            df = self.to_dataframe()
            df_main = self.to_dataframe(main=True)
            self.coords = df[["lat", "lon"]].values
            self.coords_main = df_main[["lat", "lon"]].values
            self.values = df[self.data_name].values
            self.values_main = df_main[self.data_name].values
            self.temporal_trend = self.ds.attrs["temporal_trend"]
            self.spatial_trend = df["spatial_trend"].values
            self.spatial_mean = self.ds.attrs["spatial_mean"]
            self.scale_fact = self.ds.attrs["scale_fact"]
            self.covariate_means = self.ds.attrs["covariate_means"]
            self.covariate_scales = self.ds.attrs["covariate_scales"]
            self.variance_estimate = df[self.var_name].values
            self.covariates = df[covariates]
        else:
            self.ds_main = ds.assign_coords(coords={"time": np.nan})
            df_main = self.to_dataframe(main=True)
            self.coords = self.coords_main = df_main[["x", "y"]].values
            self.values = self.values_main = df_main[self.data_name].values
        self.size = len(self.values)

    def to_dataframe(self, main: bool = False):
        """Converts the field to a data frame."""
        if main:
            return (
                self.ds_main.to_dataframe()
                .reset_index()
                .dropna(subset=[self.data_name])
            )
        else:
            return self.ds.to_dataframe().reset_index().dropna(subset=[self.data_name])

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
        type: str = "real",
    ) -> None:
        self.type = type
        self.datasets = datasets
        if type == "real":
            _check_length_match(datasets, covariates, timedeltas)
            self.timestamp = np.datetime_as_string(timestamp, unit="D")
            self.timedeltas = timedeltas
            self.covariates = covariates
            self.fields = np.array(
                [
                    Field(
                        datasets[i],
                        covariates[i],
                        self._apply_timedelta(timedeltas[i]),
                        type=type,
                    )
                    for i in range(len(datasets))
                ]
            )
        else:
            self.timestamp = np.nan
            self.timedeltas = [np.nan, np.nan]
            self.fields = np.array(
                [
                    Field(datasets[i], None, np.nan, type=type)
                    for i in range(len(datasets))
                ]
            )
        self.n_procs = len(self.fields)
        self.n_data = self._count_data()

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
        return distance_matrix(*coord_list, units=units, fast_dist=fast_dist)

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


def get_group_ids(group: pd.DataFrame):
    """Returns the group ids as a tuple (i, j)."""
    i = group.index.get_level_values("i")[0]
    j = group.index.get_level_values("j")[0]
    return i, j


def _median_abs_dev(x: np.ndarray) -> float:
    """Returns the median absolute deviation of the input array.

    Assumes the array is Normally distributed. See https://en.wikipedia.org/wiki/Median_absolute_deviation for details."""
    k = 1.4826  # scale factor assuming a normal distribution
    return k * np.nanmedian(np.abs(x - np.nanmedian(x)))


def fit_linear_trend(da: DataArray) -> DataArray:
    """Computes the monthly average of all spatial locations, and removes the trend fit by a linear model."""
    x = da.mean(dim=["lat", "lon"])
    trend = simple_linear_regression(x.values)
    return DataArray(trend, dims=["time"], coords={"time": da.time})


def fit_ols(ds: Dataset, data_name: str, covar_names: list):
    """Fit and predict the mean surface using ordinary least squares with standarized covariates. Keep track of the standardization statistics."""
    df = (
        ds.to_dataframe()
        .drop(columns=["time", f"{data_name}_var"])
        .dropna(subset=[data_name])
        .reset_index()
    )
    if df.shape[0] == 0:  # no data
        return ds[data_name] * np.nan

    means = df[covar_names].mean(axis=0, skipna=True).values
    scales = df[covar_names].std(axis=0, skipna=True).values
    covariates = df[covar_names].copy()
    for i, covar in enumerate(covar_names):
        covariates[covar] = (covariates[covar] - means[i]) / scales[i]

    model = LinearRegression().fit(covariates, df[data_name])
    df = df[["lon", "lat"]]
    df["ols_mean"] = model.predict(covariates)
    ds_pred = (
        df.set_index(["lon", "lat"])
        .to_xarray()
        .assign_coords(coords={"time": ds[data_name].time})
    )
    return ds_pred["ols_mean"], model, means, scales


def distance_matrix(
    X1: np.ndarray, X2: np.ndarray, units: str = "km", fast_dist: bool = False
) -> np.ndarray:
    """
    Computes the geodesic (or great circle if fast_dist=True) distance among all pairs of points given two sets of coordinates.
    Wrapper for scipy.spatial.distance.cdist using geopy.distance.geodesic as a the metric.

    NOTE:
    - points should be formatted in rows as [lat, lon]
    - if fast_dist=True, units are kilometers regardless of specification
    """
    # enforce 2d array in case of single point
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    if fast_dist:
        # great circle distances in kilometers
        X1_r = np.radians(X1)
        X2_r = np.radians(X2)
        return haversine_distances(X1_r, X2_r) * EARTH_RADIUS
    elif units is not None:
        # geodesic distances in specified units
        return cdist(X1, X2, lambda s_i, s_j: getattr(geodesic(s_i, s_j), units))
    else:
        # Euclidean distance
        return cdist(X1, X2)


def _preprocess_ds(ds: Dataset, timestamp: str, covariates: list) -> Dataset:
    """Apply data transformations and compute surface mean and standard deviation."""
    data_name, _ = _get_field_names(ds)
    ds_copy = ds.copy()

    # Remove linear trend over time
    ds_copy["temporal_trend"] = fit_linear_trend(ds_copy[data_name])
    ds_copy[data_name] = ds_copy[data_name] - ds_copy["temporal_trend"]

    # Select data at timestamp (fields are spatial only)
    ds_field = ds_copy.sel(time=timestamp)
    ds_field.attrs["temporal_trend"] = ds_field["temporal_trend"].values

    # Remove the spatial trend by OLS
    (
        ds_field["spatial_trend"],
        ds_field.attrs["spatial_model"],
        ds_field.attrs["covariate_means"],
        ds_field.attrs["covariate_scales"],
    ) = fit_ols(ds_field, data_name, covariates)
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
