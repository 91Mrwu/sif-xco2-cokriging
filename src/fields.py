from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
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
        timedelta=0.0,
        full_detrend=False,
        spatial_mean="constant",
        scale_fact=None,
        local_std=False,
    ):
        self.timestamp = timestamp
        self.timedelta = timedelta

        self.data_name, self.var_name = krige_tools.get_field_names(ds)
        ds_prep = krige_tools.preprocess_ds(
            ds.copy(),
            timestamp,
            full_detrend=full_detrend,
            spatial_mean=spatial_mean,
            scale_fact=scale_fact,
            local_std=local_std,
        )
        df = (
            ds_prep.sel(time=timestamp)
            .to_dataframe()
            .reset_index()
            .dropna(subset=[self.data_name])
        )
        self.ds = ds_prep
        self.coords = df[["lat", "lon"]].values
        self.values = df[self.data_name].values
        # self.temporal_mean = df["temporal_mean"].values
        # self.temporal_std = df["temporal_std"].values
        self.spatial_mean = df["spatial_mean"].values
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
        # NOTE: assumes data is already mean-zero
        df = self.ds[self.data_name].to_dataframe().dropna().reset_index()
        # Assign location and time IDs
        df["loc_id"] = df.groupby(["lat", "lon"]).ngroup()
        df["t_id"] = df.groupby(["time"]).ngroup()
        return df


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
        spatial_mean="constant",
        scale_facts=[None, None],
        local_std=False,
        dist_units="km",
        fast_dist=False,
    ):
        self.timestamp = timestamp
        self.timedelta = timedelta
        self.dist_units = dist_units
        self.fast_dist = fast_dist
        self.ds_1 = ds_1
        self.ds_2 = ds_2
        self.field_1 = Field(
            ds_1,
            timestamp,
            full_detrend=full_detrend,
            spatial_mean=spatial_mean,
            scale_fact=scale_facts[0],
            local_std=local_std,
        )
        self.field_2 = Field(
            ds_2,
            self._apply_timedelta(),
            timedelta=timedelta,
            full_detrend=full_detrend,
            spatial_mean=spatial_mean,
            scale_fact=scale_facts[1],
            local_std=local_std,
        )
        self.joint_data_vec = np.hstack((self.field_1.values, self.field_2.values))
        # self.joint_std_inverse = np.float_power(
        #     np.hstack((self.field_1.temporal_std, self.field_2.temporal_std)), -1
        # )

    def _apply_timedelta(self):
        """Returns timestamp with months offset by timedelta as string."""
        t0 = datetime.strptime(self.timestamp, "%Y-%m-%d")
        return (t0 + relativedelta(months=self.timedelta)).strftime("%Y-%m-%d")

    def get_joint_dists(self):
        """Computes block distance matrices and returns the blocks in a dict."""
        off_diag = krige_tools.distance_matrix(
            self.field_1.coords,
            self.field_2.coords,
            units=self.dist_units,
            fast_dist=self.fast_dist,
        )
        return {
            "block_11": krige_tools.distance_matrix(
                self.field_1.coords,
                self.field_1.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
            "block_12": off_diag,
            "block_21": off_diag.T,
            "block_22": krige_tools.distance_matrix(
                self.field_2.coords,
                self.field_2.coords,
                units=self.dist_units,
                fast_dist=self.fast_dist,
            ),
        }
