from datetime import datetime
from dateutil.relativedelta import relativedelta

from krige_tools import preprocess_da


class Field:
    """
    Stores data values and coordinates for a single process at a fixed time in a data frame.
    """

    def __init__(self, da, timestamp):
        df = da.sel(time=timestamp).to_dataframe().reset_index().dropna()
        self.timestamp = datetime.strptime(timestamp, "%Y-%m-%d")
        self.coords = df[["lat", "lon"]].values
        self.values = df.values[:, -1]


class MultiField:
    """
    Main class; home for data and predictions.
    """

    def __init__(
        self, da_1, da_2, timestamp, timedelta=0, detrend=False, standardize=False
    ):
        da_1 = preprocess_da(da_1, detrend=detrend, standardize=standardize)
        da_2 = preprocess_da(da_2, detrend=detrend, standardize=standardize)

        self.timestamp = timestamp
        self.timedelta = timedelta
        self.field_1 = Field(da_1, timestamp)
        self.field_2 = Field(da_2, self._apply_timedelta())

    def _apply_timedelta(self):
        """Returns timestamp with months offset by timedelta as string."""
        t0 = datetime.strptime(self.timestamp, "%Y-%m-%d")
        return (t0 + relativedelta(months=self.timedelta)).strftime("%Y-%m-%d")
