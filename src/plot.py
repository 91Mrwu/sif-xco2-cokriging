# Custom plotting wrappers
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm

from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# import cartopy.mpl.ticker as cticker
from cmcrameri import cm

from data_utils import set_main_coords, get_main_coords, get_iterable
from fields import MultiField
from model import FittedVariogram

# Global settings
XCO2_COLOR = "#4C72B0"
SIF_COLOR = "#55A868"
LINEWIDTH = 4
ALPHA = 0.6


def plot_samples(ds, cmap=cm.roma_r, title=None, fontsize=12, filename=None):
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    axes = dict(zip(ds.data_vars.keys(), [ax1, ax2]))
    cbar_ax = fig.add_subplot(gs[2])

    values = ds.to_dataframe().values
    vmax = np.max(np.abs([np.nanmin(values), np.nanmax(values)]))
    for name, da in ds.data_vars.items():
        ax = axes[name]
        im = xr.plot.imshow(
            da.T,
            vmax=vmax,
            center=0,
            cmap=cmap,
            add_colorbar=False,
            ax=ax,
        )
        ax.set_xlabel("d1", fontsize=fontsize)
        ax.set_ylabel("d2", fontsize=fontsize)
        ax.set_title(name, fontsize=fontsize)
    fig.colorbar(im, cax=cbar_ax, label="Simulated values")
    if title:
        fig.suptitle(title, size=fontsize)

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=180)


def plot_sim_pred(ds, vmax=None, robust=False, title=None, fontsize=12, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    xr.plot.imshow(
        ds["pred"].T,
        cmap=cm.roma_r,
        vmax=vmax,
        center=0,
        robust=robust,
        ax=axes[0],
        cbar_kwargs={"label": ""},
    )
    xr.plot.imshow(
        ds["pred_err"].T,
        cmap=cm.lajolla_r,
        vmin=0,
        robust=robust,
        ax=axes[1],
        cbar_kwargs={"label": ""},
    )
    names = ["Predictions", "Prediction Standard Errors"]
    for i, name in enumerate(names):
        axes[i].set_xlabel("d1", fontsize=fontsize)
        axes[i].set_ylabel("d2", fontsize=fontsize)
        axes[i].set_title(name, fontsize=fontsize)

    if title:
        fig.suptitle(title, size=fontsize)

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=180)


def prep_axes(ax, extents):
    ax.add_feature(cfeature.OCEAN)
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


def plot_da(
    da,
    vmin=None,
    vmax=None,
    robust=False,
    cmap=cm.bamako_r,
    title=None,
    label=None,
    fontsize=12,
    filename=None,
):
    PROJ = ccrs.PlateCarree()
    extents = [-130, -60, 18, 60]
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": PROJ})
    xr.plot.imshow(
        darray=da.T,
        transform=ccrs.PlateCarree(),
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        cbar_kwargs={"label": label},
    )
    prep_axes(ax, extents)
    ax.set_title(title, fontsize=fontsize)
    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=180)


def plot_df(
    df,
    data_name,
    vmin=None,
    vmax=None,
    cmap=cm.bamako_r,
    s=2,
    title=None,
    label=None,
    fontsize=12,
    filename=None,
):
    PROJ = ccrs.PlateCarree()
    extents = [-130, -60, 18, 60]
    cmap = cmap.copy()
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": PROJ})
    prep_axes(ax, extents)
    plt.scatter(
        x=df.lon,
        y=df.lat,
        c=df[data_name],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        s=s,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
    )
    cmap.set_bad(color="red")
    plt.colorbar(label=label)
    ax.set_title(title, fontsize=fontsize)
    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=180)


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


def get_data(field):
    da = field.ds_main[field.data_name].rename({"lon": "Longitude", "lat": "Latitude"})
    return da.T


def get_month_from_string(timestamp: str) -> str:
    m = np.datetime64(timestamp)
    return np.datetime_as_string(m, unit="M")


def plot_fields(
    mf: MultiField,
    data_names: list[str],
    title: str = None,
    fontsize: int = 12,
    filename: str = None,
):
    PROJ = ccrs.PlateCarree()
    CMAP = cm.roma_r
    extents = [-130, -60, 18, 60]
    if title is None:
        title = f"{','.join(data_names)}: 4x5-degree monthly average residuals"

    fig, axes = plt.subplots(
        1, mf.n_procs, figsize=(8 * mf.n_procs, 4), subplot_kw={"projection": PROJ}
    )
    fig.suptitle(title, size=fontsize)

    for i, ax in enumerate(get_iterable(axes)):
        prep_axes(ax, extents)
        xr.plot.imshow(
            darray=get_data(mf.fields[i]),
            transform=ccrs.PlateCarree(),
            ax=ax,
            cmap=CMAP,
            vmin=-3,
            center=0,
            cbar_kwargs={"label": "Standardized residuals"},
        )
        ax.set_title(
            f"{data_names[i]}: {get_month_from_string(mf.fields[i].timestamp)}",
            fontsize=fontsize,
        )

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=180)


def plot_empirical_group(ids, group, fit_result, data_names, ax):
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
        ax[idx].set_title(
            f"Cross-{fit_result.config.kind.lower()}: {data_names[ids[0]]} vs"
            f" {data_names[ids[1]]} at"
            f" {np.abs(fit_result.timedeltas[idx])} month(s) lag",
            fontsize=fit_result.fontsize,
        )
    else:
        ax[idx].set_ylabel(fit_result.scale_lab, fontsize=fit_result.fontsize)
        ax[idx].set_title(
            f"{fit_result.config.kind}: {data_names[ids[1]]}",
            fontsize=fit_result.fontsize,
        )
        if fit_result.scale_lab.lower() != "covariance":
            ax[idx].set_ylim(bottom=0.0)
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


def triangular_number(n):
    return n * (n + 1) // 2


def plot_variograms(
    fit_result: FittedVariogram,
    data_names: list[str],
    title: str = None,
    fontsize: int = 12,
    filename: str = None,
):
    # TODO: add bar chart (or number) indicating bin count (see scikit-gstat for example)
    # TODO: provide parameters in table
    n_procs = fit_result.config.n_procs
    n_plots = triangular_number(n_procs)
    fit_result.fontsize = fontsize
    if fit_result.config.kind == "Covariogram":
        fit_result.scale_lab = "Covariance"
    else:
        fit_result.scale_lab = "Semivariance"

    fig, ax = plt.subplots(
        1, n_plots, figsize=(6 * n_plots, 5), constrained_layout=True
    )

    groups1 = fit_result.df_empirical.groupby(level=[0, 1])
    for ids, df_group in groups1:
        plot_empirical_group(ids, df_group, fit_result, data_names, get_iterable(ax))

    groups2 = fit_result.df_theoretical.groupby(level=[0, 1])
    for ids, df_group in groups2:
        plot_model_group(ids, df_group, get_iterable(ax))

    if title is None:
        fig.suptitle(
            f"{fit_result.config.kind}s and cross-{fit_result.config.kind.lower()} for"
            f" {' and '.join(data_names)} residuals\n {fit_result.timestamp},"
            f" 4x5-degree North America, {fit_result.config.n_bins} bins, CompWLS:"
            f" {np.int(fit_result.cost)}",
            fontsize=fontsize,
            y=1.1,
        )
    else:
        fig.suptitle(
            title,
            fontsize=fontsize,
            y=1.1,
        )

    if filename:
        fig.savefig(f"../plots/{filename}.png", dpi=100)
