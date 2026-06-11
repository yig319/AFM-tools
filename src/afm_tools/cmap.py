"""Colormap helpers used by AFM/PFM plotting code."""

from __future__ import annotations

import matplotlib.pyplot as plt


AFM_CMAPS = {
    "height": "viridis",
    "amplitude": "viridis",
    "phase": "twilight",
    "latphase": "twilight",
}


# Look up a matplotlib colormap by AFM/PFM channel label (height→viridis, phase→twilight).
def get_afm_cmap(channel_label: str | None = None, default: str = "viridis"):
    """Return a matplotlib colormap suited to a common AFM/PFM channel label."""
    label = (channel_label or "").replace(" ", "").lower()
    name = AFM_CMAPS.get(label, default)
    return plt.get_cmap(name)


__all__ = ["AFM_CMAPS", "get_afm_cmap"]
