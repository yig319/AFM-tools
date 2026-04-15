"""Small 3D plotting helpers for AFM height maps."""

from __future__ import annotations

import numpy as np


def plot_surface_3d(height, scan_size=None, ax=None, cmap: str = "viridis", stride: int = 1):
    """Plot a height map as a 3D surface and return ``(fig, ax)``.

    Parameters
    ----------
    height:
        2D height array. Values are plotted as provided.
    scan_size:
        Optional physical scan size for x/y axes. When omitted, pixel indices
        are used.
    ax:
        Existing 3D axis. If omitted, a new figure and 3D axis are created.
    """
    import matplotlib.pyplot as plt

    z = np.asarray(height, dtype=float)
    rows, cols = z.shape
    if scan_size is None:
        x = np.arange(cols)
        y = np.arange(rows)
    else:
        x = np.linspace(0, float(scan_size), cols)
        y = np.linspace(0, float(scan_size), rows)
    xx, yy = np.meshgrid(x, y)

    if ax is None:
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot_surface(xx[::stride, ::stride], yy[::stride, ::stride], z[::stride, ::stride], cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("height")
    return fig, ax


__all__ = ["plot_surface_3d"]
