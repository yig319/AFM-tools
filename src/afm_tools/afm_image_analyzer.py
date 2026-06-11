"""Numerical AFM/PFM analysis helpers.

These functions avoid plotting decisions except for quick diagnostics. The
high-level, PLD-style preview figures live in :mod:`afm_tools.afm_viz`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sci_viz_utils.figures import imshow_percentile, layout_fig, show_images

from afm_tools.afm_utils import get_channel, load_ibw


# Fit a 2D polynomial surface z(x,y) of degree (kx,ky) via least squares.
def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    """Two dimensional polynomial fitting by least squares."""
    x, y = np.meshgrid(x, y)
    coeffs = np.ones((kx + 1, ky + 1))
    a = np.zeros((coeffs.size, x.size))

    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)


# Fit and subtract a 2D polynomial background from an AFM image, with optional debug plot.
def fit_background(img: np.ndarray, degrees=(3, 3), viz: bool = False):
    """Fit and subtract a 2D polynomial background."""
    import matplotlib.pyplot as plt

    img = np.asarray(img, dtype=float)
    x, y = np.array(range(img.shape[0])), np.array(range(img.shape[1]))
    out = polyfit2d(x, y, img, kx=degrees[0], ky=degrees[1])
    background = np.polynomial.polynomial.polygrid2d(x, y, out[0].reshape((degrees[0] + 1, degrees[1] + 1)))
    flattened = img - background

    if viz:
        fig, axes = layout_fig(3, mod=3, figsize=(9, 2.5))
        axes = np.asarray(axes).ravel()
        for ax, data, title in zip(axes, [img, background, flattened], ["original", "background", "flattened"]):
            im = imshow_percentile(ax, data, colorbar=False)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()
    return flattened, background


# Subtract a fitted 1st- or 2nd-order polynomial plane from an AFM height image.
def flatten_plane(image: np.ndarray, order: int = 1) -> np.ndarray:
    """Subtract a fitted plane or low-order polynomial background."""
    image = np.asarray(image, dtype=float)
    y, x = np.indices(image.shape)
    terms = [np.ones(image.size), x.ravel(), y.ravel()]
    if order >= 2:
        terms.extend([(x * x).ravel(), (x * y).ravel(), (y * y).ravel()])
    design = np.vstack(terms).T
    mask = np.isfinite(image.ravel())
    coeffs, *_ = np.linalg.lstsq(design[mask], image.ravel()[mask], rcond=None)
    background = (design @ coeffs).reshape(image.shape)
    return image - background


# Replace outlier pixels (surface particles/dust) with the image mean, using std thresholding.
def remove_surface_particles(img: np.ndarray, threshold: float = 3, viz: bool = False) -> np.ndarray:
    """Replace outlier pixels with the image mean."""
    import matplotlib.pyplot as plt

    img = np.asarray(img, dtype=float)
    mean, std = np.nanmean(img), np.nanstd(img)
    out = np.copy(img)
    if std == 0 or not np.isfinite(std):
        return out
    out[out < mean - threshold * std] = mean
    out[out > mean + threshold * std] = mean

    if viz:
        fig, axes = layout_fig(3, mod=3, figsize=(9, 2.5))
        axes = np.asarray(axes).ravel()
        for ax, data, title in zip(axes, [img, img - out, out], ["original", "particles", "cleaned"]):
            im = imshow_percentile(ax, data, colorbar=False)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()
    return out


# Replace statistical outlier pixels (>> N std from median) with the median value.
def remove_outliers(image: np.ndarray, threshold: float = 4.0) -> np.ndarray:
    """Replace pixels farther than ``threshold`` standard deviations with the median."""
    image = np.asarray(image, dtype=float)
    out = image.copy()
    center = np.nanmedian(out)
    scale = np.nanstd(out)
    if not np.isfinite(scale) or scale == 0:
        return out
    mask = np.abs(out - center) > threshold * scale
    out[mask] = center
    return out


# Compute root-mean-square (RMS / Sq) roughness of finite pixels in a height array.
def afm_RMS_roughness(height: np.ndarray) -> float:
    """Return root-mean-square roughness of finite pixels in the input array."""
    height = np.asarray(height, dtype=float)
    height = height[np.isfinite(height)]
    if height.size == 0:
        return np.nan
    avg = np.mean(height)
    return float(np.sqrt(np.mean((height - avg) ** 2)))


# Alias for afm_RMS_roughness — kept for backward compatibility.
def rms_roughness(height: np.ndarray) -> float:
    """Alias for :func:`afm_RMS_roughness`."""
    return afm_RMS_roughness(height)


# Extract a nearest-pixel line profile between two (row, col) points in an image.
def calculate_height_profile(image: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]):
    """Return a nearest-pixel line profile between two `(row, col)` points."""
    image = np.asarray(image)
    rows = np.linspace(p0[0], p1[0], int(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) + 1)
    cols = np.linspace(p0[1], p1[1], len(rows))
    rr = np.clip(np.round(rows).astype(int), 0, image.shape[0] - 1)
    cc = np.clip(np.round(cols).astype(int), 0, image.shape[1] - 1)
    return np.arange(len(rr)), image[rr, cc]


# Alias for calculate_height_profile — kept for backward compatibility.
def line_profile(image: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]):
    """Alias for :func:`calculate_height_profile`."""
    return calculate_height_profile(image, p0, p1)


# Compute the fraction of pixels above a threshold (Otsu or user-supplied) — used for PFM domain ratio.
def domain_fraction(image: np.ndarray, threshold: float | None = None):
    """Return positive-domain fraction, binary mask, and threshold."""
    image = np.asarray(image, dtype=float)
    if threshold is None:
        try:
            from skimage.filters import threshold_otsu

            threshold = float(threshold_otsu(image[np.isfinite(image)]))
        except Exception:
            threshold = float(np.nanmedian(image))
    mask = image > threshold
    return float(np.nanmean(mask)), mask, threshold


# Compute RMS roughness for a batch of IBW files, returning a pandas DataFrame.
def roughness_summary(files, channel_index: int = 0, flatten: bool = True, sample_parser=None) -> pd.DataFrame:
    """Compute RMS roughness for a list of IBW files."""
    rows: list[dict[str, object]] = []
    for path in files:
        path = Path(path)
        row: dict[str, object] = {"sample": sample_parser(path) if sample_parser else path.stem, "path": str(path)}
        try:
            img = load_ibw(path)
            height = get_channel(img, index=channel_index)
            height = remove_outliers(height)
            if flatten:
                height = flatten_plane(height)
            rms_m = rms_roughness(height)
            row.update(
                {
                    "sample": sample_parser(path) if sample_parser else img.sample,
                    "scan_size_m": img.scan_size_m,
                    "rms_m": rms_m,
                    "rms_nm": rms_m * 1e9,
                    "channels": ", ".join(img.labels),
                    "error": "",
                }
            )
        except Exception as exc:
            row.update({"rms_m": np.nan, "rms_nm": np.nan, "channels": "", "error": str(exc)})
        rows.append(row)
    return pd.DataFrame(rows)


# Quick-look grid plot of AFM/PFM channels — thin redirect to afm_viz.plot_afm_channels.
def plot_channels(image, max_channels: int = 6, cmap: str = "viridis"):
    """Quick-look plot for channels in an ``AFMImage`` — redirects to :func:`afm_viz.plot_afm_channels`.

    Kept for backward compatibility. All new code should call
    ``afm_tools.afm_viz.plot_afm_channels`` directly.
    """
    from .afm_viz import plot_afm_channels

    return plot_afm_channels(image, max_channels=max_channels, n_cols=max_channels, cmap=cmap, scalebar=False, show_metric_overlay=False)
