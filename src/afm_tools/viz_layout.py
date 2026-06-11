"""Compatibility aliases for generic plotting helpers.

AFM-specific plotting lives in :mod:`afm_tools.afm_viz`. Generic layout and
scale-bar primitives are owned by :mod:`sci_viz_utils.figures`.

This module re-exports:

- ``layout_fig`` -- flexible grid of matplotlib axes with auto-sizing.
- ``scalebar``  -- draw a scale-bar patch and label on an image axis.
"""

from sci_viz_utils.figures import layout_fig, scalebar

__all__ = ["layout_fig", "scalebar"]
