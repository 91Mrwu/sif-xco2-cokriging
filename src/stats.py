# Statistical functions and wrappers
import numpy as np
import xarray


def compute_xcov_1d(v1, v2, lag):
    """
    Empirical cross-covariance in 1-dimension
    Cressie and Wikle, Eq 5.4, single point.
    Parameters:
        - v1, v2: 1-d numpy arrays
        - lag: integer lag
    Returns:
        - xcov: float
    """
    # truncate appropriate end of each vector to apply lag
    x = (v1 - np.nanmean(v1))[lag:]
    y = (v2 - np.nanmean(v2))[:-lag]

    return np.nanmean(x * y)


def compute_xcov_nd(Z1, Z2, lag):
    """
    Empirical cross-covariance broadcasted over an array
    Cressie and Wikle, Eq 5.4, single location.
    Parameters:
        - Z1, Z2: 3-d numpy arrays
        - lag: integer lag
    Returns:
        - xcov: 2-d numpy array
    """
    # apply mask for nan values
    Z1_m = np.ma.array(Z1, mask=np.isnan(Z1))
    Z2_m = np.ma.array(Z2, mask=np.isnan(Z2))

    # truncate along time dim at appropriate position to apply lag
    X = (Z1_m - Z1_m.mean(axis=-1, keepdims=True))[:, :, lag:]
    Y = (Z2_m - Z2_m.mean(axis=-1, keepdims=True))[:, :, :-lag]

    # compute cross-covariance along the time dimension
    xcov = np.mean(X * Y, axis=-1)

    # return data values with missing entries filled as nan
    # NOTE: maybe use a special flag here instead
    return np.ma.filled(xcov.astype(float), np.nan)


def apply_cross_covariance(X, Y, lag=0):
    return xarray.apply_ufunc(
        compute_xcov_nd,
        X,
        Y,
        kwargs={"lag": lag},
        input_core_dims=[["time"], ["time"]],
        dask="parallelized",
        # vectorize = True, # needed if using 1-d cross-cov func
        output_dtypes=[float],
    )
