from datetime import datetime
from dateutil.relativedelta import relativedelta

import krige_tools


class Field:
    """
    Stores data values and coordinates for a single process at a fixed time in a data frame.
    """

    def __init__(
        self, ds, timestamp, detrend=False, standardize=False, scale_fact=None
    ):
        data_name, var_name = krige_tools.get_field_names(ds)
        ds = krige_tools.preprocess_ds(
            ds, detrend=detrend, standardize=standardize, scale_fact=scale_fact
        )

        df = ds.sel(time=timestamp).to_dataframe().reset_index().dropna()
        self.timestamp = datetime.strptime(timestamp, "%Y-%m-%d")
        self.coords = df[["lat", "lon"]].values
        self.values = df[data_name].values
        self.mean = df["mean"].values
        self.std = df["std"].values
        self.variance_estimate = df[var_name].values


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
        standardize=False,
        scale_fact=(None, None),
    ):
        self.timestamp = timestamp
        self.timedelta = timedelta
        self.field_1 = Field(
            ds_1,
            timestamp,
            detrend=detrend,
            standardize=standardize,
            scale_fact=scale_fact[0],
        )
        self.field_2 = Field(
            ds_2,
            self._apply_timedelta(),
            detrend=detrend,
            standardize=standardize,
            scale_fact=scale_fact[1],
        )

    def _apply_timedelta(self):
        """Returns timestamp with months offset by timedelta as string."""
        t0 = datetime.strptime(self.timestamp, "%Y-%m-%d")
        return (t0 + relativedelta(months=self.timedelta)).strftime("%Y-%m-%d")
