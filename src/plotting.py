# Custom plotting wrappers
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import xarray


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
    xarray.plot.imshow(
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
    ax.coastlines()
    ax.set_title(title, size=14)
    return (ax, cbar_ax)
