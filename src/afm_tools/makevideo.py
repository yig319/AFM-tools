"""Video export helpers for AFM/PFM frame stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_video(frames, output, fps: int = 10, cmap: str = "gray") -> Path:
    """Write a 3D frame stack to a video file with imageio.

    ``frames`` should have shape ``(n_frames, rows, cols)``. Values are
    percentile-normalized to 8-bit before writing so raw AFM-like arrays can be
    previewed directly.
    """
    import imageio.v3 as iio
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3:
        raise ValueError("frames must be a 3D array: (n_frames, rows, cols)")

    lo, hi = np.nanpercentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    colormap = plt.get_cmap(cmap)
    rgb = (colormap(norm)[..., :3] * 255).astype(np.uint8)
    iio.imwrite(output, rgb, fps=fps)
    return output


make_video = write_video


__all__ = ["make_video", "write_video"]
