# Custom plotting wrappers
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Global settings
XCO2_COLOR = "#4C72B0"
SIF_COLOR = "#55A868"
LINEWIDTH = 4
ALPHA = 0.6


def set_gridlines(ax):
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=0.8,
        color="black",
        alpha=0.5,
        linestyle="--",
        draw_labels=True,
    )
    gl.top_labels = False
    gl.left_labels = True
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True
    gl.xlocator = mticker.FixedLocator([-120, -100, -80, -60])
    gl.ylocator = mticker.FixedLocator([30, 50])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def plot_da(
    da,
    title="",
    ax=None,
    cbar_ax=None,
    cmap="jet",
    cbar_kwargs=None,
    vmin=None,
    vmax=None,
    robust=True,
    add_colorbar=True,
):
    """
    Wrapper to create an image / raster plot of a data array in a given axis.
    """
    ax.set_global()
    xr.plot.imshow(
        darray=da.T,
        transform=ccrs.PlateCarree(),
        ax=ax,
        cbar_ax=cbar_ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar_kwargs=cbar_kwargs,
        robust=robust,
        add_colorbar=add_colorbar,
    )
    ax.add_feature(cfeature.OCEAN, zorder=9)
    ax.coastlines(zorder=10)
    ax.set_title(title, size=14)
    return (ax, cbar_ax)


def qq_plots(mf):
    # NOTE: assumes XCO2 is field_1
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    sm.qqplot(mf.field_1.values, line="45", ax=axes[0])
    axes[0].set_title(f"XCO$_2$: {mf.field_1.timestamp}", fontsize=12)

    sm.qqplot(mf.field_2.values, line="45", ax=axes[1])
    axes[1].set_title(f"SIF: {mf.field_2.timestamp}", fontsize=12)

    fig.suptitle("Q-Q plots: 4x5-degree residuals over North America", fontsize=14)


def resid_climatology(df, title, filename=None):
    """Paired residual climatology from dataframe with time and 2 value columns."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["time"], df["sif"], color=SIF_COLOR, lw=LINEWIDTH, alpha=ALPHA, label="SIF"
    )
    ax.plot(
        df["time"],
        df["xco2"],
        color=XCO2_COLOR,
        lw=LINEWIDTH,
        alpha=ALPHA,
        label="XCO$_2$",
        zorder=10,
    )

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="upper left", fontsize=12)

    ax.set_ylabel("Average process residuals", size=12)
    ax.set_xlabel("Time", size=12)
    ax.set_title(title, size=12)

    plt.tight_layout()
    if filename:
        plt.savefig(f"../plots/{filename}.png", dpi=200)


def resid_coord_avg(mf, axes=None, filename=None):
    """Subplots of averages over each coordinate (both processes).
    NOTE: assumes field_1 is XCO2 and field_2 is SIF
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    x_lat = np.unique(mf.field_1.coords[:, 0])
    x_lon = np.unique(mf.field_1.coords[:, 1])

    axes[0].set_title("Residual average over longitude", size=12)
    axes[0].set_ylabel("Average process residuals", size=12)
    axes[0].set_xlabel("Latitude", size=12)
    axes[0].plot(
        x_lat,
        mf.field_1.ds.xco2.mean(dim="lon").values,
        color=XCO2_COLOR,
        lw=LINEWIDTH,
        alpha=ALPHA,
        label="XCO$_2$",
        zorder=10,
    )
    axes[0].plot(
        x_lat,
        mf.field_2.ds.sif.mean(dim="lon").values,
        color=SIF_COLOR,
        lw=LINEWIDTH,
        alpha=ALPHA,
        label="SIF",
    )

    axes[1].set_title("Residual average over latitude", size=12)
    if axes is not None:
        axes[1].set_ylabel("Average process residuals", size=12)
    axes[1].set_xlabel("Longitude", size=12)
    axes[1].plot(
        x_lon,
        mf.field_1.ds.xco2.mean(dim="lat").values,
        color=XCO2_COLOR,
        lw=LINEWIDTH,
        alpha=ALPHA,
        label="XCO$_2$",
        zorder=10,
    )
    axes[1].plot(
        x_lon,
        mf.field_2.ds.sif.mean(dim="lat").values,
        color=SIF_COLOR,
        lw=LINEWIDTH,
        alpha=ALPHA,
        label="SIF",
    )

    for ax in axes:
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.legend(loc="upper left", fontsize=12)

    if axes is None:
        plt.tight_layout()
    if filename:
        plt.savefig(f"../plots/{filename}.png", dpi=200)


def plot_fields(mf, coord_avg=False, filename=None):
    title = "XCO$_2$ and SIF: 4x5-degree monthly averages\n Temporal trend and spatial mean surface removed; residuals scaled by spatial standard deviation"

    da_xco2 = mf.field_1.ds.xco2
    da_sif = mf.field_2.ds.sif
    extents = [-125, -60, 18, 60]

    if coord_avg:
        # fig, f_axs = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
        # gs = f_axs[0, 0].get_gridspec()
        # for ax in f_axs[0, 0:]:
        #     ax.remove()
        # ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.EqualEarth())
        # ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.EqualEarth())
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(80, 100)
        ax1 = fig.add_subplot(gs[0:50, 0:49], projection=ccrs.EqualEarth())
        ax2 = fig.add_subplot(gs[0:50, 51:100], projection=ccrs.EqualEarth())
        axes = [fig.add_subplot(gs[55:80, 0:39]), fig.add_subplot(gs[55:80, 51:90])]
        fig.suptitle(title, size=14, y=0.95)
    else:
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(100, 100)
        ax1 = fig.add_subplot(gs[:, 0:49], projection=ccrs.EqualEarth())
        ax2 = fig.add_subplot(gs[:, 51:100], projection=ccrs.EqualEarth())
        fig.suptitle(title, size=14, y=1.01)

    xr.plot.imshow(
        darray=da_xco2.T,
        transform=ccrs.PlateCarree(),
        ax=ax1,
        cmap="jet",
        vmin=-2,
        center=0,
        cbar_kwargs={"label": "Process residuals"},
    )
    ax1.coastlines()
    ax1.set_extent(extents)
    ax1.set_title(
        f"XCO$_2$: {pd.to_datetime(da_xco2.time.values).strftime('%Y-%m')}", fontsize=14
    )

    xr.plot.imshow(
        darray=da_sif.T,
        transform=ccrs.PlateCarree(),
        ax=ax2,
        cmap="jet",
        vmin=-2,
        center=0,
        cbar_kwargs={"label": "Process residuals"},
    )
    ax2.coastlines()
    ax2.set_extent(extents)
    ax2.set_title(
        f"SIF: {pd.to_datetime(da_sif.time.values).strftime('%Y-%m')}", fontsize=14
    )
    for ax in [ax1, ax2]:
        set_gridlines(ax)

    if coord_avg:
        # resid_coord_avg(mf, f_axs[1, 0:])
        resid_coord_avg(mf, axes)

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=200)


def param_labels(params, cross=False):
    p = np.round_(params, decimals=3)
    if cross:
        return f"nu: {p[0]}\nlen_scale: {p[1]}\nrho: {p[2]}"
    else:
        return f"sigma: {p[0]}\n nu: {p[1]}\n len_scale: {p[2]}\n nugget: {p[3]}"


def plot_semivariograms(vario_res, timestamp, timedelta, filename=None):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True, sharex=True)
    ax[0, 1].axis("off")

    lags = vario_res["xco2"]["bin_center"].values
    bin_width = lags[2] - lags[1]

    for i, var in enumerate(["xco2", "sif"]):
        df = vario_res[var]
        df.plot(
            x="bin_center",
            y="bin_mean",
            kind="scatter",
            color="white",
            ax=ax[i, i],
            label="Empirical semivariogram",
        )
        for j, txt in enumerate(df["count"]):
            ax[i, i].annotate(
                np.int(txt),
                (df.bin_center.values[j], df.bin_mean.values[j]),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        ax[i, i].set_title(var, fontsize=12)
        ax[i, i].set_ylabel("Semivariance", fontsize=12)
        ax[i, i].set_xlabel("Seperation distance (km)", fontsize=12)
        ax[i, i].legend(loc="upper left")

    df = vario_res["xco2:sif"]
    df.plot(
        x="bin_center",
        y="bin_mean",
        kind="scatter",
        color="white",
        ax=ax[1, 0],
        label="Empirical cross-semivariogram",
    )
    for j, txt in enumerate(df["count"]):
        ax[1, 0].annotate(
            np.int(txt),
            (df.bin_center.values[j], df.bin_mean.values[j]),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax[1, 0].set_ylabel("Semivariance", fontsize=12)
    ax[1, 0].set_xlabel("Seperation distance (km)", fontsize=12)

    ax[0, 0].set_title("Semivariogram: XCO$_2$", fontsize=12)
    ax[1, 0].set_title(
        f"Cross-semivariogram: XCO$_2$ vs SIF at {np.abs(timedelta)} month(s) lag",
        fontsize=12,
    )
    ax[1, 1].set_title("Semivariogram: SIF", fontsize=12)

    fig.suptitle(
        f"Semivariograms and cross-semivariogram for XCO$_2$ and SIF residuals\n {timestamp}, 4x5-degree North America, bin width {np.int(bin_width)} km",
        fontsize=14,
    )

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=100)


def plot_covariograms(covario_res, timestamp, timedelta, filename=None):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True, sharex=True)
    ax[0, 1].axis("off")

    lags = covario_res["xco2"]["bin_center"].values
    bin_width = lags[2] - lags[1]

    for i, var in enumerate(["xco2", "sif"]):
        df = covario_res[var]
        df.plot(
            x="bin_center",
            y="bin_mean",
            kind="scatter",
            color="white",
            ax=ax[i, i],
            label="Empirical covariogram",
        )
        for j, txt in enumerate(df["count"]):
            ax[i, i].annotate(
                np.int(txt),
                (df.bin_center.values[j], df.bin_mean.values[j]),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        ax[i, i].set_title(var, fontsize=12)
        ax[i, i].set_ylabel("Covariance", fontsize=12)
        ax[i, i].set_xlabel("Seperation distance (km)", fontsize=12)
        ax[i, i].legend()

    df = covario_res["xco2:sif"]
    df.plot(
        x="bin_center",
        y="bin_mean",
        kind="scatter",
        color="white",
        ax=ax[1, 0],
        label="Empirical cross-covariogram",
    )
    for j, txt in enumerate(df["count"]):
        ax[1, 0].annotate(
            np.int(txt),
            (df.bin_center.values[j], df.bin_mean.values[j]),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax[1, 0].legend(loc="upper left")
    ax[1, 0].set_ylabel("Cross-covariance", fontsize=12)
    ax[1, 0].set_xlabel("Seperation distance (km)", fontsize=12)

    ax[0, 0].set_title("Covariogram: XCO$_2$", fontsize=12)
    ax[1, 0].set_title(
        f"Cross-covariogram: XCO$_2$ vs SIF at {np.abs(timedelta)} month(s) lag",
        fontsize=12,
    )
    ax[1, 1].set_title("Covariogram: SIF", fontsize=12)

    fig.suptitle(
        f"Covariograms and cross-covariogram for XCO$_2$ and SIF residuals\n {timestamp}, 4x5-degree North America, bin width {np.int(bin_width)} km",
        fontsize=14,
    )

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=100)

