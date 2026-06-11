"""Compatibility aliases for generic video writing helpers.

Re-exports from :mod:`sci_viz_utils.video`:

- ``write_video`` -- write a numpy image stack to a video file.
- ``make_video``   -- alias for ``write_video``.
"""

from sci_viz_utils.video import write_video

make_video = write_video

__all__ = ["make_video", "write_video"]
