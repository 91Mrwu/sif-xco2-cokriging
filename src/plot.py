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

# import cartopy.mpl.ticker as cticker
from cmcrameri import cm

from data_utils import set_main_coords, get_main_coords
from model import FittedVariogram

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

    fig.suptitle("Q-Q plots: 4x5-degree residuals over North America", fontsize=12)


def raw_climatology(df, title, filename=None):
    # Plot global daily climatology
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax_r1 = ax.twinx()

    # Line plots
    ax_r1.scatter(
        df["time"], df["sif"], color=SIF_COLOR, s=20, alpha=ALPHA, label="SIF"
    )
    ax.scatter(
        df["time"],
        df["xco2"],
        color=XCO2_COLOR,
        s=20,
        alpha=ALPHA,
        label="XCO$_2$",
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


def get_data(field, lon_centers, lat_centers):
    da = (
        get_main_coords(field.ds, lon_centers, lat_centers)
        .sel(time=field.timestamp)[field.data_name]
        .rename({"lon": "Longitude", "lat": "Latitude"})
    )
    return da.T


def plot_fields(mf, coord_avg=False, filename=None):
    # title = "XCO$_2$ and SIF: 4x5-degree monthly averages\n Temporal trend and spatial mean surface removed; residuals scaled by spatial standard deviation"
    # title = "XCO$_2$ and SIF: 4x5-degree monthly average residuals\n Temporal trend and spatial mean surface removed; residuals scaled by spatial median absolute deviation"
    PROJ = ccrs.PlateCarree()
    CMAP = cm.roma.reversed()
    title = "XCO$_2$ and SIF: 4x5-degree monthly average residuals"

    extents = [-130, -60, 18, 60]
    lon_centers, lat_centers = set_main_coords()

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
        fig.suptitle(title, size=12, y=0.95)
    else:
        # fig = plt.figure(figsize=(20, 5))
        # gs = fig.add_gridspec(100, 100)
        # ax1 = fig.add_subplot(gs[:, 0:52], projection=PROJ)
        # ax2 = fig.add_subplot(gs[:, 48:100], projection=PROJ)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 4), subplot_kw={"projection": PROJ}
        )
        fig.suptitle(title, size=12)

    xr.plot.imshow(
        darray=get_data(mf.fields[0], lon_centers, lat_centers),
        transform=ccrs.PlateCarree(),
        ax=ax1,
        cmap=CMAP,
        vmin=-3,
        center=0,
        cbar_kwargs={"label": "Process residuals"},
    )
    xr.plot.imshow(
        darray=get_data(mf.fields[1], lon_centers, lat_centers),
        transform=ccrs.PlateCarree(),
        ax=ax2,
        cmap=CMAP,
        vmin=-3,
        center=0,
        cbar_kwargs={"label": "Process residuals"},
    )

    for ax in [ax1, ax2]:
        # set_gridlines_new(ax)
        prep_axes(ax, extents)

    ax1.set_title(f"XCO$_2$: {mf.fields[0].timestamp}", fontsize=12)
    ax2.set_title(f"SIF: {mf.fields[1].timestamp}", fontsize=12)

    if coord_avg:
        # resid_coord_avg(mf, f_axs[1, 0:])
        resid_coord_avg(mf, axes)

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=180)


def plot_empirical_group(ids, group, fit_result, ax):
    idx = np.sum(ids)
    group.plot(
        x="bin_center",
        y="bin_mean",
        kind="scatter",
        color="black",
        alpha=0.8,
        ax=ax[idx],
        label=f"Empirical {fit_result.config.kind.lower()}",
    )
    if idx == 1:
        ax[idx].set_ylabel(
            f"Cross-{fit_result.scale_lab.lower()}", fontsize=fit_result.fontsize
        )
    else:
        ax[idx].set_ylabel(fit_result.scale_lab, fontsize=fit_result.fontsize)
        if fit_result.scale_lab.lower() != "covariance":
            ax[idx].set_ylim(bottom=0)
    ax[idx].set_xlabel(
        f"Separation distance ({fit_result.config.dist_units})",
        fontsize=fit_result.fontsize,
    )
    ax[idx].legend()


def plot_model_group(ids, group, ax):
    idx = np.sum(ids)
    ax[idx].plot(
        group["distance"],
        group["variogram"],
        linestyle="--",
        color="black",
        label="Fitted model",
    )


def plot_variograms(
    fit_result: FittedVariogram,
    data_names: list[str],
    fontsize: int = 12,
    filename: str = None,
):
    # TODO: add bar chart (or number) indicating bin count
    # TODO: provide parameters in table
    fit_result.fontsize = fontsize
    if fit_result.config.kind == "Covariogram":
        fit_result.scale_lab = "Covariance"
    else:
        fit_result.scale_lab = "Semivariance"

    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    groups1 = fit_result.df_empirical.groupby(level=[0, 1])
    for ids, df_group in groups1:
        plot_empirical_group(ids, df_group, fit_result, ax)

    groups2 = fit_result.df_theoretical.groupby(level=[0, 1])
    for ids, df_group in groups2:
        plot_model_group(ids, df_group, ax)

    kind = fit_result.config.kind
    ax[0].set_title(f"{kind}: {data_names[0]}", fontsize=fontsize)
    ax[1].set_title(
        f"Cross-{kind.lower()}: {data_names[0]} vs {data_names[1]} at"
        f" {np.abs(fit_result.timedeltas[1])} month(s) lag",
        fontsize=fontsize,
    )
    ax[2].set_title(f"{kind}: {data_names[1]}", fontsize=fontsize)

    fig.suptitle(
        f"{kind}s and cross-{kind.lower()} for {data_names[0]} and"
        f" {data_names[1]} residuals\n {fit_result.timestamp}, 4x5-degree North"
        f" America, {fit_result.config.n_bins} bins, CompWLS:"
        f" {np.int(fit_result.cost)}",
        fontsize=fontsize,
        y=1.1,
    )

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=100)
