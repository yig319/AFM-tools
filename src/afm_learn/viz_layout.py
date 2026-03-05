"""Local plotting layout helpers used by AFM visualization utilities.

This module replaces the previous dependency on ``m3util.viz.layout`` by
providing only the functions used inside this package.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path, patches, patheffects
from matplotlib.gridspec import GridSpec

Path = path.Path
PathPatch = patches.PathPatch


def layout_fig(
    graph,
    mod=None,
    figsize=None,
    subplot_style="subplots",
    spacing=(0.3, 0.3),
    parent_ax=None,
    layout="compressed",
    **kwargs,
):
    """Create a figure with either ``plt.subplots`` or ``GridSpec`` layout."""
    if mod is None:
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        else:
            mod = 7

    nrows = graph // mod + (graph % mod > 0)
    wspace, hspace = spacing

    if parent_ax is None:
        if figsize is None:
            figsize = (3 * mod, 3 * nrows)
        elif isinstance(figsize, tuple) and (figsize[0] is None or figsize[1] is None):
            w, h = figsize
            unit_w = kwargs.pop("unit_w", 3)
            unit_h = kwargs.pop("unit_h", 3)
            w = w if w is not None else unit_w * mod
            h = h if h is not None else unit_h * nrows
            figsize = (w, h)

        if subplot_style == "gridspec":
            fig = plt.figure(figsize=figsize)
            width_ratios = kwargs.pop("width_ratios", [1] * mod)
            height_ratios = kwargs.pop("height_ratios", [1] * nrows)
            gs = GridSpec(
                nrows,
                mod,
                figure=fig,
                width_ratios=width_ratios,
                height_ratios=height_ratios,
                wspace=wspace,
                hspace=hspace,
            )
            axes = [fig.add_subplot(gs[i // mod, i % mod]) for i in range(graph)]
        elif subplot_style == "subplots":
            if layout is None:
                fig, axes = plt.subplots(nrows, mod, figsize=figsize)
            else:
                fig, axes = plt.subplots(nrows, mod, figsize=figsize, layout=layout)
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()[:graph]
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
        else:
            raise ValueError("Invalid layout option. Choose 'gridspec' or 'subplots'.")

        return fig, axes

    width_ratios = kwargs.pop("width_ratios", [1] * mod)
    height_ratios = kwargs.pop("height_ratios", [1] * nrows)
    bbox = parent_ax.get_position()
    fig = parent_ax.figure
    gs = GridSpec(
        nrows,
        mod,
        figure=fig,
        left=bbox.x0,
        bottom=bbox.y0,
        right=bbox.x1,
        top=bbox.y1,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=wspace,
        hspace=hspace,
    )
    axes = [fig.add_subplot(gs[i // mod, i % mod]) for i in range(graph)]
    if len(axes) == 1:
        axes = axes[0]
    return None, axes


def _path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):
    """Create and add a rectangular path patch to an axis."""
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices = [
        (locations[0], locations[2]),
        (locations[1], locations[2]),
        (locations[1], locations[3]),
        (locations[0], locations[3]),
        (0, 0),
    ]
    vertices = np.array(vertices, float)
    rect_path = Path(vertices, codes)
    pathpatch = PathPatch(
        rect_path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        ls=linestyle,
        lw=lineweight,
    )
    axes.add_patch(pathpatch)


def scalebar(axes, image_size, scale_size, units="nm", loc="br", text_fontsize=7):
    """Add a scale bar to an axis displaying image data."""
    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    fract = scale_size / image_size
    x_point = np.linspace(x_lim[0], x_lim[1], np.int64(np.floor(image_size)))

    height_fraction = 0.03 * (y_lim[1] - y_lim[0])
    width_offset = 0.05 * (x_lim[1] - x_lim[0])

    if loc == "br":
        x_start = x_point[np.int64(0.9 * image_size // 1)] - width_offset
        x_end = x_point[np.int64((0.9 - fract) * image_size // 1)] - width_offset
        y_start = y_lim[0] + 0.1 * (y_lim[1] - y_lim[0])
        y_end = y_start + height_fraction
        y_label_height = y_start + 3 * height_fraction
    elif loc == "tr":
        x_start = x_point[np.int64(0.9 * image_size // 1)] - width_offset
        x_end = x_point[np.int64((0.9 - fract) * image_size // 1)] - width_offset
        y_start = y_lim[1] - 0.1 * (y_lim[1] - y_lim[0])
        y_end = y_start - height_fraction
        y_label_height = y_start - 3.5 * height_fraction
    else:
        raise ValueError("loc must be 'br' or 'tr'")

    _path_maker(axes, [x_start, x_end, y_start, y_end], "w", "k", "-", 0.25)
    axes.text(
        (x_start + x_end) / 2,
        y_label_height,
        f"{scale_size} {units}",
        size=text_fontsize,
        weight="bold",
        ha="center",
        va="center",
        color="w",
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="k")],
    )
