from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

import spatial_tools


class Field:
    """
    Stores data values and coordinates for a single process at a fixed time in a data frame.
    """

    def __init__(self, ds, timestamp, covariates=None):
        self.timestamp = timestamp
        self.data_name, self.var_name = _get_field_names(ds)
        self.ds = preprocess_ds(ds, timestamp)
        df = self.ds.to_dataframe().reset_index().dropna(subset=[self.data_name])
        self.coords = df[["lat", "lon"]].values
        self.values = df[self.data_name].values
        self.temporal_trend = self.ds.attrs["temporal_trend"]
        self.spatial_trend = df["spatial_trend"].values
        self.spatial_mean = self.ds.attrs["spatial_mean"]
        self.scale_fact = self.ds.attrs["scale_fact"]
        self.variance_estimate = df[self.var_name].values
        if covariates is not None:
            self.covariates = df[covariates]

    def to_xarray(self):
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
    Stores a bivariate process, each of class Field, along with modelling attributes.
    """

    def __init__(
        self,
        ds_1,
        ds_2,
        timestamp,
        timedelta=0,
        dist_units="km",
        fast_dist=False,
        covars_1=None,
        covars_2=None,
    ):
        self.timestamp = np.datetime_as_string(timestamp, unit="D")
        self.timedelta = timedelta
        self.dist_units = dist_units
        self.fast_dist = fast_dist
        self.ds_1 = ds_1
        self.ds_2 = ds_2
        self.field_1 = Field(ds_1, timestamp, covariates=covars_1)
        self.field_2 = Field(ds_2, self._apply_timedelta(), covariates=covars_2)
        self.joint_data_vec = np.hstack((self.field_1.values, self.field_2.values))
        # self.joint_std_inverse = np.float_power(
        #     np.hstack((self.field_1.temporal_std, self.field_2.temporal_std)), -1
        # )

    def _apply_timedelta(self):
        """Returns timestamp with month offset by timedelta as string."""
        t0 = datetime.strptime(self.timestamp, "%Y-%m-%d")
        return (t0 + relativedelta(months=self.timedelta)).strftime("%Y-%m-%d")

    def get_joint_dists(self):
        """Computes block distance matrices and returns the blocks in a dict."""
        off_diag = spatial_tools.distance_matrix(
            self.field_1.coords,
            self.field_2.coords,
            units=self.dist_units,
            fast_dist=self.fast_dist,
        )
        return {
            "block_11": spatial_tools.distance_matrix(
                self.field_1.coords,
                self.field_1.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
            "block_12": off_diag,
            "block_21": off_diag.T,
            "block_22": spatial_tools.distance_matrix(
                self.field_2.coords,
                self.field_2.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
        }


def _get_field_names(ds):
    """Returns data and estimated variance names from dataset."""
    var_name = [name for name in list(ds.keys()) if "_var" in name][0]
    data_name = var_name.replace("_var", "")
    return data_name, var_name


def _median_abs_dev(x):
    """Returns the median absolute deviation of the input array.

    Assumes the array is Normally distributed. See https://en.wikipedia.org/wiki/Median_absolute_deviation for details."""
    k = 1.4826  # scale factor assuming a normal distribution
    return k * np.nanmedian(np.abs(x - np.nanmedian(x)))


def preprocess_ds(ds, timestamp):
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
    if data_name == "sif":
        covar_names = ["evi"]
    else:
        covar_names = ["lon", "lat"]
    ds_field["spatial_trend"] = spatial_tools.fit_ols(ds_field, data_name, covar_names)
    ds_field[data_name] = ds_field[data_name] - ds_field["spatial_trend"]

    # Standardize the residuals
    ds_field.attrs["spatial_mean"] = np.nanmean(ds_field[data_name].values)
    ds_field.attrs["scale_fact"] = np.nanstd(ds_field[data_name].values)
    # ds_field.attrs["scale_fact"] = _median_abs_dev(ds_field[data_name].values)
    ds_field[data_name] = (
        ds_field[data_name] - ds_field.attrs["spatial_mean"]
    ) / ds_field.attrs["scale_fact"]

    return ds_field
