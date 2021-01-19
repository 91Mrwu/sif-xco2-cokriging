# Statistical functions and wrappers
import numpy as np
import pandas as pd
import xarray
from sklearn.linear_model import LinearRegression

## Counts / replications
def get_count(da):
    """
    Count non-missing elements along the time dimension.
    Inputs:
        - da: xarray data array (lon x lat x time)
    Outputs:
        - da_counts: (lon x lat)
    """
    return np.count_nonzero(~np.isnan(da), axis=-1)


def apply_count(da):
    return xarray.apply_ufunc(
        get_count,
        da,
        input_core_dims=[["time"]],
        output_dtypes=[float],
        dask="parallelized",
    )


## Trend fitting
def detrend(x):
    """
    Fit and remove a trend from a vector with indices as the covariate. Return the detrended vector and the slope of the trend seperately.
    Inputs:
        - x: 1-d numpy array
    Outputs:
        - slope: float
        - z: 1-d numpy array
    """
    if np.isnan(x).all():
        return x, np.nan
    else:
        # obtain covariate from array indices
        data = np.stack([np.arange(x.size), x])
        data = data[:, ~np.isnan(data).any(axis=0)]

        # fit model and remove trend from non-missing elements
        X = data[0, :].reshape(-1, 1)
        y = data[1, :]
        model = LinearRegression().fit(X, y)

        z = np.copy(x)
        z[~np.isnan(z)] = y - model.predict(X)

        return z, model.coef_


def apply_detrend(da):
    return xarray.apply_ufunc(
        detrend,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"], []],
        output_dtypes=[float, float],
        dask="parallelized",
        vectorize=True,
    )


## Cross-correlation
def compute_xcor_1d(v1, v2, lag=0, tau=None):
    """
    Empirical cross-correlation in 1-dimension
    Inputs:
        - v1, v2: 1-d numpy arrays
        - lag: integer lag
        - tau: integer threshold indicating minimum number of values needed for a valid computation
    Outputs:
        - xcor: float
    """
    # use masked arrays
    v1_m = np.ma.array(v1, mask=np.isnan(v1))
    v2_m = np.ma.array(v2, mask=np.isnan(v2))

    # remove the mean along time dim
    x = v1_m - v1_m.mean()
    y = v2_m - v2_m.mean()
    if lag is not 0:
        # truncate along time dim at appropriate position to apply lag
        x = x[lag:]
        y = y[:-lag]

    if tau is not None:
        if np.count_nonzero(~np.isnan(x * y)) < tau:
            return np.nan

    xcor = np.sum(x * y) / (np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
    return np.ma.filled(xcor.astype(float), np.nan)


def compute_xcor_nd(Z1, Z2, lag=0, tau=None):
    """
    Empirical cross-correlation broadcasted over an array
    Inputs:
        - Z1, Z2: xarray data array (lon x lat x time)
        - lag: integer lag
        - tau: integer threshold indicating minimum number of values needed for a valid computation
    Outputs:
        - xcor: xarray data array (lon x lat)
    """
    # apply mask for nan values
    Z1_m = np.ma.array(Z1, mask=np.isnan(Z1))
    Z2_m = np.ma.array(Z2, mask=np.isnan(Z2))

    # remove the mean along time dim
    X = Z1_m - Z1_m.mean(axis=-1, keepdims=True)
    Y = Z2_m - Z2_m.mean(axis=-1, keepdims=True)
    if lag != 0:
        # truncate along time dim at appropriate position to apply lag
        X = X[:, :, lag:]
        Y = Y[:, :, :-lag]

    # compute cross-correlation along the time dimension
    xcor = np.sum(X * Y, axis=-1) / (
        np.sqrt(np.sum(X * X, axis=-1)) * np.sqrt(np.sum(Y * Y, axis=-1))
    )

    if tau:
        # mask out computations which use fewer than tau non-missing values
        xcor = np.ma.masked_where(
            np.count_nonzero(~np.isnan(X * Y), axis=-1) < tau, xcor
        )

    # return data values with missing entries filled as nan
    return np.ma.filled(xcor.astype(float), np.nan)


def apply_xcor(da1, da2, lag=0, tau=None):
    # remove process trend along the time dimension
    Z1, _ = apply_detrend(da1)
    Z2, _ = apply_detrend(da2)

    return xarray.apply_ufunc(
        compute_xcor_nd,
        Z1,
        Z2,
        kwargs={"lag": lag, "tau": tau},
        input_core_dims=[["time"], ["time"]],
        output_dtypes=[float],
        dask="parallelized",
    )


def optim_lag_nd(da1, da2, lag_bnds, tau=None):
    """
    Determine which lag maximizes the absolute value of the cross-correlation along the time dimension.
    Inputs:
        - da1, da2: xarray data array (lon x lat x time)
        - lag_bnds: (low, high) tuple of integer bounds
        - tau: integer threshold indicating minimum number of values needed for a valid computation
    Outputs:
        - optim_lag: xarray data array (lon x lat)
    """
    # remove process trend along the time dimension
    Z1, _ = apply_detrend(da1)
    Z2, _ = apply_detrend(da2)

    # compute an xcor field for each lag in specified range
    lags = np.arange(*lag_bnds)
    xcor_fields = []
    [
        xcor_fields.append(
            xarray.apply_ufunc(
                compute_xcor_nd,
                Z1,
                Z2,
                kwargs={"lag": lag, "tau": tau},
                input_core_dims=[["time"], ["time"]],
                output_dtypes=[float],
                dask="parallelized",
            ).values
        )
        for lag in lags
    ]

    # overlay fields along new lag axis, and compute argmax along that axis
    xcor_stack = np.stack(xcor_fields, axis=2)
    xcor_stack = np.ma.array(xcor_stack, mask=np.isnan(xcor_stack))

    # NOTE: if slice is all nans, this will return an optimal lag of 0 (ok as long as data array itself is masked)
    optim_lag = np.ma.argmax(np.abs(xcor_stack), axis=2)
    xcor = np.squeeze(
        np.take_along_axis(xcor_stack, np.expand_dims(optim_lag, axis=2), 2), axis=2
    )

    # return data set with arrays optimal lag and corresponding xcor for each element
    return xarray.Dataset(
        {"optim_lag": (["lon", "lat"], optim_lag), "xcor": (["lon", "lat"], xcor),},
        coords={"lon": da1.lon, "lat": da1.lat,},
    )


## Wrappers
def get_stats(DS):
    """
    Compute all of the above statistics for SIF and XCO2 data arrays.
    """
    DS["sif_count"] = apply_count(DS.sif)
    DS["xco2_count"] = apply_count(DS.xco2)
    sif_resid, DS["sif_slope"] = apply_detrend(DS.sif)
    xco2_resid, DS["xco2_slope"] = apply_detrend(DS.xco2)
    DS["sif_std"] = sif_resid.std(dim="time")
    DS["xco2_std"] = xco2_resid.std(dim="time")
    return DS


def get_stats_df(df_group, lags=[0], tau=None):
    """
    Compute the count, slope, std. dev., and cross-cor for SIF and XCO2 dataframe columns.
    """
    sif_resid, sif_slope = detrend(df_group["sif"].values)
    xco2_resid, xco2_slope = detrend(df_group["xco2"].values)

    df = pd.DataFrame(
        {
            "sif_count": df_group["sif"].dropna().count(),
            "xco2_count": df_group["xco2"].dropna().count(),
            "sif_slope": sif_slope,
            "xco2_slope": xco2_slope,
            "sif_std": np.nanstd(sif_resid),
            "xco2_std": np.nanstd(xco2_resid),
        }
    )

    for lag in lags:
        df[f"xcor_lag{lag}"] = compute_xcor_1d(xco2_resid, sif_resid, lag=lag, tau=tau)

    return df

