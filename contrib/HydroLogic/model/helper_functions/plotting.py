from typing import List

import contextily as ctx
import matplotlib.pyplot as plt
import rasterio
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.plot import show
import numpy as np
import matplotlib.colors

def default_plot(dfs: List, variable: str, labels: List = None) -> None:
    """Basic plot function that takes a number of DataFrames and plots the 'variable' columns in one figure

    Args:
        dfs (List): list of dataframes (can be 1) that are to be plotted. Should all contain a column with variable
        variable (str): name of variable and column to be selected from dataframes
        labels (List): labels of plotted data frames, e.g. ["Measured", "Simulated"]

    Returns:
        None
    """
    plt.figure(figsize=(12, 4))
    for ix, df in enumerate(dfs):
        if labels is None:
            plt.plot(df[variable].dropna())
        else:
            plt.plot(df[variable].dropna(), label=labels[ix])

    if labels is not None:
        plt.legend()

    plt.gca().update(dict(title=r"Plot of: " + variable, xlabel="date", ylabel=variable))
    plt.grid()
    plt.show()


def raster_plot_with_context(
    raster_path: str,
    epsg: int = 28992,
    basemap: bool = True,
    clabel: str = None,
    cmap: str = None,
    vmin: float = None,
    vmax: float = None,
    title: str = None,
    discrete_cbar: float = None,
    transparent_below_vmin: bool = False,
    custom_legend: bool = False,
    custom_colors: list = None,
    custom_labels: list = None,
    custom_bounds: list = None,
    legend_title: str = None,
    legend_title_size: int = 14,
    cmap_extend: str = 'both'
) -> plt.figure:

    """
    Function to create a plot from a tiff file

    Args:
        raster_path (str): location of tiff file
        epsg (int): EPSG identifier of coordinate reference system. Default is EPSG:28992, Dutch RDS
        basemap (bool): Whether to load a basemap or not
        clabel (str): label for colorbar
        cmap (str): string identifier for matplotlib colormap
        title (str): title of plot
        vmin (float): lower bound of colorbar
        vmax (float): upper bound of colorbar
        discrete_cbar (float): number of categories of discrete colorbar
        transparent_below_vmin (bool): set values outside colorbar transparent
        custom_legend (bool): add a custom legend with custom classes to the plot
        custom_colors (list): list of custom colors to be used
        custom_labels (list): list of custom labels to be used
        custom_bounds (list): list of class boundaries to be used
        legend_title (str): title of the legend
        legend_title_size (int): fontsize of the legend title
        cmap_extend (str): whether the bounds extend to min and max. Choose from {'both', 'min','max', 'neither'}

    Returns:
        fig (plt.Figure): matplotlib figure object
        ax (plt.axis): matplotlib axis

    """
    # Check whether the necessary elements are supplied
    if (discrete_cbar is not None) and (vmin or vmax is None): 
        raise NameError('vmin and vmax cannot be of type None when discrete_cbar is supplied')
    if custom_legend:
        if (custom_colors is None) or (custom_labels is None) or (custom_bounds is None):
            raise NameError('custom_colors, custom_labels and custom_bounds need to be supplied when custom_legend is True')
        if not len(custom_colors) == len(custom_labels):
            raise ValueError('custom_colors, custom_labels and custom_bounds need to be of the same size')

    kwargs = {}
    if not custom_legend:
        if cmap is not None:
            if discrete_cbar is not None: cmap = plt.get_cmap(cmap, discrete_cbar)
            else: cmap = plt.get_cmap(cmap)

            if transparent_below_vmin: cmap.set_under('k', alpha=0)
            kwargs["cmap"] = cmap
    else: 
        cmap = matplotlib.colors.ListedColormap(custom_colors)
        norm = matplotlib.colors.BoundaryNorm(custom_bounds, cmap.N, extend=cmap_extend)
        if transparent_below_vmin: cmap.set_under('k', alpha=0)
        kwargs["cmap"] = cmap
        kwargs["norm"] = norm

    if title is not None:
        kwargs["title"] = title
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    
    figsize = [12, 6]
    dpi = 150

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    
    src = rasterio.open(raster_path)
    show(source=src, ax=ax, zorder=1, **kwargs)
    ax.invert_yaxis()

    if not custom_legend:
        if discrete_cbar is not None:
            dist_per_color = (vmax - vmin) / (discrete_cbar)
            cbar = plt.colorbar(ax.get_images()[0], ax=ax, 
                                ticks = np.arange(vmin-0.5*dist_per_color,vmax+0.5*dist_per_color,dist_per_color),
                                extend = 'max')
        else: cbar = plt.colorbar(ax.get_images()[0], ax=ax)

        if clabel is not None:
            cbar.set_label(clabel)
    else:
        patches = [Patch(color=custom_colors[i], label=custom_labels[i])
                for i in range(len(custom_colors))]

        leg = ax.legend(handles=patches,
                bbox_to_anchor=(1.05, 1),
                facecolor="white",
                frameon = False,
                title=legend_title,
                title_fontsize=legend_title_size,
                loc='upper left')
        leg._legend_box.align = "left"

    ax.set(xlabel="x", ylabel="y")

    if basemap:
        ctx.add_basemap(ax, crs=epsg, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.add_artist(ScaleBar(1, location="lower right"))

    north_arrow(ax=ax)

    return fig, ax


def north_arrow(ax: Axes, x: float = 0.95, y: float = 0.2, arrow_length: float = 0.1) -> Axes:
    """
    Adds a north arrow to a plot in axis ax

    Args:
        ax (Axes): axes to plot north arrow on
        x (float): relative location of arrow on x-axis (0-1).
        y (float): relative location of arrow on y-axis (0-1).
        arrow_length (float): size of arrow.

    Returns:
        ax (Axes): axes on which north arrow was plotted
    """
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - arrow_length),
        arrowprops=dict(facecolor="black", width=5, headwidth=15),
        ha="center",
        va="center",
        fontsize=20,
        xycoords=ax.transAxes,
    )
    return ax
