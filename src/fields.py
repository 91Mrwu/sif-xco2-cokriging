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
        self,
        ds,
        timestamp,
        detrend=False,
        center=False,
        standardize=False,
        scale_fact=None,
    ):
        # self.timestamp = datetime.strptime(timestamp, "%Y-%m-%d")
        self.timestamp = timestamp

        data_name, var_name = krige_tools.get_field_names(ds)
        ds_prep = krige_tools.preprocess_ds(
            ds.copy(),
            detrend=detrend,
            center=center,
            standardize=standardize,
            scale_fact=scale_fact,
        )
        # TODO: add flags for types of transformations applied
        # TODO: is it possible to investigate the assumption of unbiased error?

        df = ds_prep.sel(time=timestamp).to_dataframe().reset_index().dropna()
        self.coords = df[["lat", "lon"]].values
        self.values = df[data_name].values
        self.mean = df["mean"].values
        self.std = df["std"].values
        self.variance_estimate = df[var_name].values

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
        detrend=False,
        center=False,
        standardize=False,
        scale_fact=(None, None),
    ):
        self.timestamp = timestamp
        self.timedelta = timedelta
        self.ds_1 = ds_1
        self.ds_2 = ds_2
        self.field_1 = Field(
            ds_1,
            timestamp,
            detrend=detrend,
            center=center,
            standardize=standardize,
            scale_fact=scale_fact[0],
        )
        self.field_2 = Field(
            ds_2,
            self._apply_timedelta(),
            detrend=detrend,
            center=center,
            standardize=standardize,
            scale_fact=scale_fact[1],
        )

    def _apply_timedelta(self):
        """Returns timestamp with months offset by timedelta as string."""
        t0 = datetime.strptime(self.timestamp, "%Y-%m-%d")
        return (t0 + relativedelta(months=self.timedelta)).strftime("%Y-%m-%d")
