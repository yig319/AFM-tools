"""Compatibility aliases for generic plotting helpers.

AFM-specific plotting lives in :mod:`afm_tools.afm_viz`. Generic layout and
scale-bar primitives are owned by :mod:`sci_viz_utils.figures`.
"""

from sci_viz_utils.figures import layout_fig, scalebar

__all__ = ["layout_fig", "scalebar"]
