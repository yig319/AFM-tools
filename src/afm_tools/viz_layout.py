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
    fraction: float = 0.25,
    color: str = "white",
    linewidth: float = 0,
    text_color: str | None = None,
    **kwargs,
):
    """Add a simple scale bar patch and label to an image axis."""
    valid_locs = {"br", "bl", "tr", "tl"}
    if loc not in valid_locs:
        raise ValueError(f"loc must be one of {sorted(valid_locs)}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if scale_size <= 0:
        raise ValueError("scale_size must be positive")

    text_color = text_color or color
    bar_px = max(float(image_size) * fraction, 1.0)
    bar_label = float(scale_size) * fraction
    pad = float(image_size) * 0.06
    height = max(float(image_size) * 0.015, 1.0)

    x0 = pad if "l" in loc else float(image_size) - pad - bar_px
    y0 = pad if "t" in loc else float(image_size) - pad - height

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

    label = f"{bar_label:g} {units}".strip()
    va = "bottom" if "t" in loc else "top"
    y_text = y0 + height + pad * 0.15 if "t" in loc else y0 - pad * 0.15
    ax.text(
        x0 + bar_px / 2,
        y_text,
        label,
        color=text_color,
        ha="center",
        va=va,
        fontsize=9,
    )
    return patch


__all__ = ["layout_fig", "scalebar"]
