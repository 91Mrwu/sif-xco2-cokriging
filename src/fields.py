from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import xarray as xr

import krige_tools


class Field:
    """
    Stores data values and coordinates for a single process at a fixed time in a data frame.
    """

    def __init__(
        self, ds, timestamp, full_detrend=False, standardize_window=False,
    ):
        # self.timestamp = datetime.strptime(timestamp, "%Y-%m-%d")
        self.timestamp = timestamp

        self.data_name, self.var_name = krige_tools.get_field_names(ds)
        ds_prep = krige_tools.preprocess_ds(
            ds.copy(),
            timestamp,
            full_detrend=full_detrend,
            standardize_window=standardize_window,
        )
        # TODO: add flags for types of transformations applied
        # TODO: is it possible to investigate the assumption of unbiased error?

        df = (
            ds_prep.sel(time=timestamp)
            .to_dataframe()
            .reset_index()
            .dropna(subset=[self.data_name])
        )
        self.ds = ds_prep
        self.coords = df[["lat", "lon"]].values
        self.values = df[self.data_name].values
        self.mean = df["mean"].values
        self.std = df["std"].values
        self.variance_estimate = df[self.var_name].values

    def to_xarray(self):
        """Converts the field to an xarray dataset."""
        return (
            pd.DataFrame(
                {
                    "lat": self.coords[:, 0],
                    "lon": self.coords[:, 1],
                    "values": self.values,
                }
            )
            .set_index(["lon", "lat"])
            .to_xarray()
        )

    def get_spacetime_df(self):
        """Converts the spatio-temporal dataset associated with the timestamp to a data frame."""
        return (
            self.ds.to_dataframe()
            .drop(columns=[self.var_name, "mean", "std"])
            .dropna()
            .reset_index()
        )


class MultiField:
    """
    Main class.
    """

    def __init__(
        self,
        ds_1,
        ds_2,
        timestamp,
        timedelta=0,
        full_detrend=False,
        standardize_window=False,
    ):
        self.timestamp = timestamp
        self.timedelta = timedelta
        self.ds_1 = ds_1
        self.ds_2 = ds_2
        self.field_1 = Field(
            ds_1,
            timestamp,
            full_detrend=full_detrend,
            standardize_window=standardize_window,
        )
        self.field_2 = Field(
            ds_2,
            self._apply_timedelta(),
            full_detrend=full_detrend,
            standardize_window=standardize_window,
        )

    def _apply_timedelta(self):
        """Returns timestamp with months offset by timedelta as string."""
        t0 = datetime.strptime(self.timestamp, "%Y-%m-%d")
        return (t0 + relativedelta(months=self.timedelta)).strftime("%Y-%m-%d")
