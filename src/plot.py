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
import cartopy.mpl.ticker as cticker
from cmcrameri import cm

from data_utils import set_main_coords, get_main_coords

# Global settings
XCO2_COLOR = "#4C72B0"
SIF_COLOR = "#55A868"
LINEWIDTH = 4
ALPHA = 0.6


def prep_axes(ax, extents):
    ax.coastlines()
    ax.set_extent(extents)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=0.8,
        color="black",
        alpha=0.5,
        linestyle="--",
        draw_labels=True,
    )
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True
    gl.xlocator = mticker.FixedLocator([-120, -100, -80, -60.0])
    gl.ylocator = mticker.FixedLocator([30, 50])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # gl.xlabel_style = {"size": 14}
    # gl.ylabel_style = {"size": 14}


# def set_gridlines_new(ax):
#     ax.set_xticks([-120.0, -100.0, -80.0, -60.0], crs=ccrs.PlateCarree())
#     ax.set_xticklabels([-120.0, -100.0, -80.0, -60.0])
#     ax.set_yticks([30.0, 50.0], crs=ccrs.PlateCarree())
#     ax.set_yticklabels([30.0, 50.0])

#     lon_formatter = cticker.LongitudeFormatter()
#     lat_formatter = cticker.LatitudeFormatter()
#     ax.xaxis.set_major_formatter(lon_formatter)
#     ax.yaxis.set_major_formatter(lat_formatter)
#     ax.grid(linewidth=0.8, color="black", alpha=0.5, linestyle="--")


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


def raw_climatology(df, title, filename=None):
    # Plot global daily climatology
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax_r1 = ax.twinx()

    # Line plots
    ax_r1.scatter(
        df["time"], df["sif"], color=SIF_COLOR, s=20, alpha=ALPHA, label="SIF"
    )
    ax.scatter(
        df["time"], df["xco2"], color=XCO2_COLOR, s=20, alpha=ALPHA, label="XCO$_2$",
    )
    ax.scatter([], [], color=SIF_COLOR, s=20, alpha=ALPHA, label="SIF")

    # Customize axes
    ax.tick_params(axis="y", colors=XCO2_COLOR, labelsize=12)
    ax_r1.tick_params(axis="y", colors=SIF_COLOR, labelsize=12)
    ax.yaxis.label.set_color(XCO2_COLOR)
    ax_r1.yaxis.label.set_color(SIF_COLOR)
    ax.legend(loc="upper left", fontsize=12)

    # Add titles
    ax_r1.set_ylabel("SIF 740nm [W/m$^2$/sr/$\mu$m]", size=12)
    ax.set_ylabel("XCO$_2$ [ppm]", size=12)
    ax.set_xlabel("Time", size=12)
    ax.set_title(title, size=12)
    # plt.setp(ax.spines.values(), color="white")
    # plt.setp(ax_r1.spines.values(), color="white")
    plt.tight_layout()

    if filename:
        plt.savefig(f"../plots/{filename}.png", dpi=200)


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
    # title = "XCO$_2$ and SIF: 4x5-degree monthly averages\n Temporal trend and spatial mean surface removed; residuals scaled by spatial standard deviation"
    # title = "XCO$_2$ and SIF: 4x5-degree monthly average residuals\n Temporal trend and spatial mean surface removed; residuals scaled by spatial median absolute deviation"
    PROJ = ccrs.PlateCarree()
    CMAP = cm.roma
    title = "XCO$_2$ and SIF: 4x5-degree monthly average residuals"

    extents = [-130, -60, 18, 60]
    lon_centers, lat_centers = set_main_coords()
    da_xco2 = (
        get_main_coords(mf.field_1.ds, lon_centers, lat_centers)
        .sel(time=mf.field_1.timestamp)["xco2"]
        .rename({"lon": "Longitude", "lat": "Latitude"})
    )
    da_sif = (
        get_main_coords(mf.field_2.ds, lon_centers, lat_centers)
        .sel(time=mf.field_2.timestamp)["sif"]
        .rename({"lon": "Longitude", "lat": "Latitude"})
    )

    if coord_avg:
        # fig, f_axs = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
        # gs = f_axs[0, 0].get_gridspec()
        # for ax in f_axs[0, 0:]:
        #     ax.remove()
        # ax1 = fig.add_subplot(gs[0, 0], projection=PROJ)
        # ax2 = fig.add_subplot(gs[0, 1], projection=PROJ)
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(80, 100)
        ax1 = fig.add_subplot(gs[0:50, 0:49], projection=PROJ)
        ax2 = fig.add_subplot(gs[0:50, 51:100], projection=PROJ)
        axes = [fig.add_subplot(gs[55:80, 0:39]), fig.add_subplot(gs[55:80, 51:90])]
        fig.suptitle(title, size=14, y=0.95)
    else:
        fig = plt.figure(figsize=(20, 5))
        # gs = fig.add_gridspec(100, 100)
        # ax1 = fig.add_subplot(gs[:, 0:52], projection=PROJ)
        # ax2 = fig.add_subplot(gs[:, 48:100], projection=PROJ)
        fig.suptitle(title, size=14)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, 10), subplot_kw={"projection": PROJ}
        )

    xr.plot.imshow(
        darray=da_xco2.T,
        transform=ccrs.PlateCarree(),
        ax=ax1,
        cmap=CMAP,
        vmin=-2,
        center=0,
        cbar_kwargs={"label": "Process residuals"},
    )
    xr.plot.imshow(
        darray=da_sif.T,
        transform=ccrs.PlateCarree(),
        ax=ax2,
        cmap=CMAP,
        vmin=-2,
        center=0,
        cbar_kwargs={"label": "Process residuals"},
    )

    for ax in [ax1, ax2]:
        # set_gridlines_new(ax)
        prep_axes(ax, extents)

    ax1.set_title(
        f"XCO$_2$: {pd.to_datetime(da_xco2.time.values).strftime('%Y-%m')}", fontsize=24
    )
    ax2.set_title(
        f"SIF: {pd.to_datetime(da_sif.time.values).strftime('%Y-%m')}", fontsize=24
    )

    if coord_avg:
        # resid_coord_avg(mf, f_axs[1, 0:])
        resid_coord_avg(mf, axes)

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=180)


def param_labels(params):
    p = np.round_(params, decimals=3)
    fit_1 = f"sigma: {p[0]}\n nu: {p[1]}\n len_scale: {p[2]}\n tau: {p[3]}"
    fit_2 = f"sigma: {p[7]}\n nu: {p[8]}\n len_scale: {p[9]}\n tau: {p[10]}"
    fit_cross = f"nu: {p[4]}\nlen_scale: {p[5]}\nrho: {p[6]}"
    return {"fit_1": fit_1, "fit_2": fit_2, "fit_cross": fit_cross}


def plot_model(df_fit, params, ax):
    labels = param_labels(params)
    for i, fname in enumerate(["fit_1", "fit_cross", "fit_2"]):
        ax[i].plot(
            df_fit["distance"],
            df_fit[fname],
            linestyle="--",
            color="black",
            label="Fitted model",
        )
        if i != 1:
            ax[i].text(
                0.95,
                0.05,
                labels[fname],
                transform=ax[i].transAxes,
                ha="right",
                va="bottom",
                size=14,
            )
    ax[1].text(
        0.05,
        0.05,
        labels["fit_cross"],
        transform=ax[1].transAxes,
        ha="left",
        va="bottom",
        size=14,
    )


def plot_variograms(
    res_obj,
    timestamp,
    timedelta,
    params=None,
    type_lab="Semivariogram",
    scale_lab="Semivariance",
    filename=None,
):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    lags = res_obj["xco2"]["bin_center"].values
    bin_width = lags[2] - lags[1]

    for i, var in enumerate(["xco2", "xco2:sif", "sif"]):
        df = res_obj[var]
        df.plot(
            x="bin_center",
            y="bin_mean",
            kind="scatter",
            color="black",
            alpha=0.8,
            ax=ax[i],
            label=f"Empirical {type_lab.lower()}",
        )
        if i == 1:
            ax[i].set_ylabel(f"Cross-{scale_lab.lower()}", fontsize=14)
        else:
            ax[i].set_ylabel(scale_lab, fontsize=14)
            if scale_lab.lower() != "covariance":
                ax[i].set_ylim(bottom=0)
        ax[i].set_title(var, fontsize=14)
        ax[i].set_xlabel("Separation distance (km)", fontsize=14)
        ax[i].legend()
        ax[i].tick_params(axis="both", which="major", labelsize=12)

    if "fit" in res_obj.keys() and params is not None:
        plot_model(res_obj["fit"], params, ax)

    ax[0].set_title(f"{type_lab}: XCO$_2$", fontsize=14)
    ax[1].set_title(
        f"Cross-{type_lab.lower()}: XCO$_2$ vs SIF at {np.abs(timedelta)} month(s) lag",
        fontsize=14,
    )
    ax[2].set_title(f"{type_lab}: SIF", fontsize=14)

    fig.suptitle(
        f"{type_lab}s and cross-{type_lab.lower()} for XCO$_2$ and SIF residuals\n {timestamp}, 4x5-degree North America, bin width {np.int(bin_width)} km",
        fontsize=14,
    )

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=100)

