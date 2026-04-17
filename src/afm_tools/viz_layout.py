"""Small plotting layout helpers for AFM/PFM visualizations."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def layout_fig(graph: int, mod: int = 3, subplot_style: str = "subplots", figsize=None, **kwargs):
    """Create a figure with ``graph`` visible axes arranged in rows of ``mod``."""
    if graph < 1:
        raise ValueError("graph must be at least 1")
    if mod < 1:
        raise ValueError("mod must be at least 1")

    rows = math.ceil(graph / mod)
    if figsize is None:
        figsize = (4 * min(mod, graph), 3.5 * rows)

    if subplot_style not in {"subplots", "gridspec"}:
        raise ValueError("subplot_style must be 'subplots' or 'gridspec'")

    fig, axes = plt.subplots(rows, mod, figsize=figsize, squeeze=False, **kwargs)
    flat_axes = list(axes.ravel())
    for ax in flat_axes[graph:]:
        fig.delaxes(ax)
    return fig, flat_axes[:graph]


def scalebar(
    ax,
    image_size: float,
    scale_size: float,
    units: str = "",
    loc: str = "br",
    pixel_size: float | None = None,
    color: str = "white",
    linewidth: float = 0,
    text_color: str | None = None,
    text_offset: float = 0.35,
    text_position: str = "above",
    **kwargs,
):
    """Add a scale bar patch and label to an image axis.

    ``image_size`` and ``scale_size`` are physical lengths in ``units``. This
    restores the original AFM visualizer behavior where ``scale_size`` is the
    label shown on the bar, such as ``500 nm``, ``1 µm``, ``2 µm`` or ``5 µm``.
    ``pixel_size`` is the image width in pixels; when omitted, ``image_size`` is
    also used as the drawing width for simple synthetic tests.
    """
    valid_locs = {"br", "bl", "tr", "tl"}
    if loc not in valid_locs:
        raise ValueError(f"loc must be one of {sorted(valid_locs)}")
    if text_position not in {"above", "below"}:
        raise ValueError("text_position must be 'above' or 'below'")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if scale_size <= 0:
        raise ValueError("scale_size must be positive")

    text_fontsize = kwargs.pop("text_fontsize", kwargs.pop("fontsize", 9))
    text_color = text_color or color
    pixel_size = float(pixel_size or image_size)
    bar_px = max(pixel_size * float(scale_size) / float(image_size), 1.0)
    pad = pixel_size * 0.06
    height = max(pixel_size * 0.015, 1.0)

    x0 = pad if "l" in loc else pixel_size - pad - bar_px
    y0 = pad if "t" in loc else pixel_size - pad - height

    patch = Rectangle(
        (x0, y0),
        bar_px,
        height,
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
        **kwargs,
    )
    ax.add_patch(patch)

    label = f"{float(scale_size):g} {units}".strip()
    if text_position == "above":
        y_text = y0 - pad * text_offset
        va = "bottom"
    else:
        y_text = y0 + height + pad * text_offset
        va = "top"
    ax.text(
        x0 + bar_px / 2,
        y_text,
        label,
        color=text_color,
        ha="center",
        va=va,
        fontsize=text_fontsize,
    )
    return patch


__all__ = ["layout_fig", "scalebar"]
