# Utilities for reading, writing, and formatting data
import warnings
from collections import Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from xarray import Dataset, open_dataset
from regionmask.defined_regions import natural_earth


## Misc
def get_iterable(x):
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)


## Reading
def prep_sif(ds: Dataset) -> Dataset:
    """Preprocess an OCO-2 SIF Lite file.

    TODO:
    - add check for extremely large observation error and remove those data values in preprocessing

    NOTE:
    SIF_Uncertainty_740nm is defined as "estimated 1-sigma uncertainty of Solar Induced Fluorescence at 740 nm. Uncertainty computed from continuum level radiance at 740 nm." Squaring this value yeilds the variance of the measurement error which will be added to the diagonals of the covariance matrix.
    """

    # drop unused variables
    variable_list = [
        "Daily_SIF_740nm",
        "SIF_Uncertainty_740nm",
        "Quality_Flag",
        "Longitude",
        "Latitude",
        "Delta_Time",
    ]
    ds = ds[variable_list]

    # apply quality filters
    ds["SIF_plus_3sig"] = ds.Daily_SIF_740nm + 3 * ds.SIF_Uncertainty_740nm
    ds = ds.where(ds.Quality_Flag != 2, drop=True)
    ds = ds.where(ds.SIF_plus_3sig > 0, drop=True)

    # format dataset
    return Dataset(
        {
            "sif": (["time"], ds.Daily_SIF_740nm),
            "sif_var": (["time"], ds.SIF_Uncertainty_740nm ** 2),
        },
        coords={
            "lon": (["time"], ds.Longitude),
            "lat": (["time"], ds.Latitude),
            "time": ds.Delta_Time.values,
        },
    )


def prep_xco2(ds: Dataset) -> Dataset:
    """Preprocess an OCO-2 FP Lite file.

    TODO:
    - add check for extremely large observation error and remove those data values in preprocessing

    NOTE:
    xco2_uncertainty is defined as "the posterior uncertainty in XCO2 calculated by the L2 algorithm, in ppm. This is generally 30-50% smaller than the true retrieval uncertainty." Doubling this value yields a conservative estimate of the variance of the measurement error which will be added to the diagonals of the covariance matrix.
    """

    # drop unused variables
    variable_list = [
        "xco2",
        "xco2_uncertainty",
        "xco2_quality_flag",
        "longitude",
        "latitude",
        "time",
    ]
    ds = ds[variable_list]

    # apply quality filters
    ds = ds.where(ds.xco2_quality_flag == 0, drop=True)

    # format dataset
    return Dataset(
        {"xco2": (["time"], ds.xco2), "xco2_var": (["time"], ds.xco2_uncertainty * 2)},
        coords={
            "lon": (["time"], ds.longitude),
            "lat": (["time"], ds.latitude),
            "time": ds.time.values,
        },
    )


def prep_evi(ds: Dataset) -> Dataset:
    """Preprocess a MODIS EVI dataset."""
    data_name = "CMG 0.05 Deg Monthly EVI"
    extents = [-130, 18, -60, 62]  # [minx, miny, maxx, maxy]
    ds_clip = ds.rio.clip_box(*extents)
    return Dataset(
        {"evi": (["lon", "lat"], ds_clip[data_name].squeeze().T.values)},
        coords={
            "lon": (["lon"], ds_clip.x.values),
            "lat": (["lat"], ds_clip.y.values),
            "time": datetime.fromisoformat(ds_clip.RANGEBEGINNINGDATE),
        },
    )


def read_transcom(path: str) -> Dataset:
    """
    Read 1-degree TransCom 3 region map into xarray dataset.
    """
    ds = open_dataset(path)
    ds = ds.where(ds.region < 12, drop=True)
    ds = ds.where(ds.region != 0, drop=True)
    return ds


## Wrangling
class GridConfig:
    def __init__(
        self,
        extents: tuple = None,
        lon_res: float = 1,
        lat_res: float = 1,
        lon_offset: float = 0,
        lat_offset: float = 0,
    ) -> None:
        if not (lon_offset == 0 or lat_offset == 0):
            raise ValueError("`lon_offset` and/or `lat_offset` must be zero")
        if extents is None:
            extents = (-180, 180, -90, 90)
        else:
            self.extents = extents
        self.lon_res = lon_res
        self.lat_res = lat_res
        self.lon_offset = lon_offset
        self.lat_offset = lat_offset
        self.lon_bounds = _prep_bounds(extents[:2], lon_res, lon_offset)
        self.lat_bounds = _prep_bounds(extents[2:], lat_res, lat_offset)


class SpatialGrid:
    def __init__(self, config: GridConfig) -> None:
        """Establish longitude and latitude bins and centerpoints on a spatial grid."""
        self.config = config
        self.lon_bins, self.lon_centers = _prep_bins(config.lon_bounds, config.lon_res)
        self.lat_bins, self.lat_centers = _prep_bins(config.lat_bounds, config.lat_res)

    def bounds_check(self, df: pd.DataFrame) -> bool:
        if not (
            self.lon_bins.min() <= df.lon.min()
            and self.lon_bins.max() >= df.lon.max()
            and self.lat_bins.min() <= df.lat.min()
            and self.lat_bins.max() >= df.lat.max()
        ):
            warnings.warn(
                "Dataset coordinates not within grid extents; may produce unexpected"
                f" behavior: ({df.lon.min()}, {df.lon.max()}, {df.lat.min()},"
                f" {df.lat.max()})"
            )


def _prep_bounds(bounds: tuple, res: float, offset: float) -> tuple:
    """Adjust bounds based on resolution and offset. Returns as (lwr, upr)."""
    half_res = 0.5 * res * np.array([-1, 1])
    bounds = np.array(bounds) + half_res + offset
    return tuple(bounds)


def _prep_bins(bounds: tuple, res: float) -> np.ndarray:
    edges = np.arange(bounds[0], bounds[1] + res, res)
    centers = (edges[1:] + edges[:-1]) / 2
    return edges, centers


def regrid(
    ds: Dataset = None, df: pd.DataFrame = None, config: GridConfig = None
) -> pd.DataFrame:
    """
    Convert dataset to dataframe and assign coordinates using a regular grid.
    """
    if ds is not None:
        df = ds.to_dataframe().reset_index()
    elif df is None:
        warnings.warn("No data provided.")

    if config is None:
        config = GridConfig()
    grid = SpatialGrid(config)
    grid.bounds_check(df)

    # overwrite lon-lat values with grid values
    df["lon"] = pd.cut(df.lon, grid.lon_bins, labels=grid.lon_centers).astype(float)
    df["lat"] = pd.cut(df.lat, grid.lat_bins, labels=grid.lat_centers).astype(float)
    return df


def land_grid(config: GridConfig = None) -> pd.DataFrame:
    """Collect land locations on a regular grid as an array. Returns rows with entries [[lat, lon]]."""
    # establish a fine resolution grid of 0.25 degrees for accuracy
    config_fine = GridConfig(config.extents, lon_res=0.25, lat_res=0.25)
    grid_fine = SpatialGrid(config_fine)
    land = natural_earth.land_110
    mask = land.mask(grid_fine.lon_centers, grid_fine.lat_centers)
    # regrid to desired resolution and remove non-land areas
    df_mask = (
        regrid(ds=mask, config=config)
        .dropna(subset=["region"])
        .groupby(["lon", "lat"])
        .mean()  # mean of binaries just reassigns value
        .reset_index()
    )
    return df_mask[["lat", "lon"]].assign(land=lambda x: 1).set_index(["lon", "lat"])


def monthly_avg(df_grid: pd.DataFrame) -> pd.DataFrame:
    """Group dataframe by relabeled lat-lon coordinates and compute monthy average."""
    return (
        df_grid.groupby(["lon", "lat"])
        .resample("1MS", on="time")
        .mean()
        .drop(columns=["lon", "lat"])
        .reset_index()
    )


def apply_land_mask(df: pd.DataFrame, config: GridConfig = None) -> pd.DataFrame:
    df_land = land_grid(config)
    return (
        df.join(df_land, on=["lon", "lat"], how="outer")
        .dropna(subset=["land"])
        .reset_index()
        .drop(columns=["land", "index"])
    )


def prep_gridded_df(
    ds: Dataset, config: GridConfig, aggregate: bool = True
) -> pd.DataFrame:
    """Aggregate irregular data to a regular grid of monthly averages within the specified extents (land only). Return as data frame."""
    df = ds.to_dataframe().reset_index()
    bounds = (
        (df.lon >= config.lon_bounds[0])
        & (df.lon <= config.lon_bounds[1])
        & (df.lat >= config.lat_bounds[0])
        & (df.lat <= config.lat_bounds[1])
    )
    # drop data outside domain extents so it's not included in edge bin averages
    df = df.loc[bounds].reset_index()
    df_grid = regrid(df=df, config=config)
    if "index" in df_grid.columns:
        df_grid = df_grid.drop(columns="index")
    if aggregate:
        df_grid = monthly_avg(df_grid)
    return apply_land_mask(df_grid, config)


def augment_dataset(ds: Dataset) -> pd.DataFrame:
    """Prepare gridded dataframes for each longitude and latitude offset, and return as a single dataframe."""
    extents = (-125, -65, 22, 58)
    lat_offsets = np.linspace(-1.5, 2, 8)
    lon_offsets = np.linspace(-2, 2.5, 10)
    # drop zero offset from one set so there is no repeat of the base coordinates
    lon_offsets = lon_offsets[lon_offsets != 0]

    config_list_lat = [
        GridConfig(extents=extents, lon_res=5, lat_res=4, lat_offset=lat_off)
        for lat_off in lat_offsets
    ]
    config_list_lon = [
        GridConfig(extents=extents, lon_res=5, lat_res=4, lon_offset=lon_off)
        for lon_off in lon_offsets
    ]
    frame_list_lat = [prep_gridded_df(ds, config) for config in config_list_lat]
    frame_list_lon = [prep_gridded_df(ds, config) for config in config_list_lon]
    return pd.concat(frame_list_lat + frame_list_lon)


def set_main_coords(
    extents: tuple = None, lon_res: float = 5, lat_res: float = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Sets the base coordinates for augmentation reference."""
    if extents is None:
        extents = (-125, -65, 22, 58)
    config = GridConfig(extents, lon_res=lon_res, lat_res=lat_res)
    grid = SpatialGrid(config)
    return grid.lon_centers, grid.lat_centers


def get_main_coords(
    ds: Dataset, lon_centers: np.ndarray = None, lat_centers: np.ndarray = None
) -> Dataset:
    """Returns the dataset with base longitudinal coordinates only."""
    if lon_centers is None or lat_centers is None:
        lon_centers, lat_centers = set_main_coords()
    return (
        ds.to_dataframe()
        .reset_index()
        .merge(pd.DataFrame({"lat": lat_centers}), on="lat", how="inner")
        .merge(pd.DataFrame({"lon": lon_centers}), on="lon", how="inner")
        .set_index(["lon", "lat", "time"])
        .to_xarray()
    )


def produce_climatology_conus(ds: Dataset, freq: str) -> pd.DataFrame:
    extents = (-125, -65, 22, 58)
    config = GridConfig(extents, lon_res=5, lat_res=4)
    return (
        prep_gridded_df(ds, config, aggregate=False)
        .dropna(subset=["lon", "lat"])
        .drop(columns=["lon", "lat"])
        .groupby(pd.Grouper(key="time", freq=freq))
        .mean()
        .reset_index()
    )


# def map_transcom(ds: Dataset, ds_tc: Dataset) -> pd.DataFrame:
#     """
#     Regrid dataset to 1-degree grid and merge TransCom regions.
#     """
#     # regrid the dataset to 1-degree
#     df_grid = regrid(ds, lon_res=1, lat_res=1).dropna().reset_index()

#     # get transcom
#     df_regions = ds_tc.to_dataframe().dropna().reset_index()

#     # merge, format, return
#     return (
#         df_grid.merge(df_regions, on=["lon", "lat"], how="inner")
#         .drop(columns=["lon", "lat"])
#         .dropna()
#         .set_index(["time"])
#     )


def to_xarray(coords: np.ndarray, **kwargs) -> Dataset:
    """Format data variables as xarray data array or dataset.

    NOTE: coords must be formatted in rows as [[lat, lon]].
    """
    return (
        pd.DataFrame({**{"lat": coords[:, 0], "lon": coords[:, 1]}, **kwargs})
        .set_index(["lon", "lat"])
        .to_xarray()
    )
