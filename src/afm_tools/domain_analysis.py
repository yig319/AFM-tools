from __future__ import annotations

import numpy as np

from afm_tools.afm_image_analyzer import domain_fraction


# Find the dominant histogram peaks in an AFM/PFM image (e.g. for phase domain separation).
def find_histogram_peaks(
    image,
    bins: int = 256,
    num_peaks: int = 2,
    distance="auto",
    threshold_factor: float = 1.5,
    min_prominence: float = 5,
    debug: bool = False,
):
    """Find dominant peaks in the histogram of an AFM/PFM image."""
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    values = np.asarray(image, dtype=float).ravel()
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    if distance == "auto":
        distance = max(1, bins // max(4, num_peaks * 4))
    prominence = max(float(min_prominence), np.std(counts) * threshold_factor)
    peaks, props = find_peaks(counts, distance=distance, prominence=prominence)
    if len(peaks) > num_peaks:
        order = np.argsort(counts[peaks])[-num_peaks:]
        peaks = peaks[order]

    peaks = peaks[np.argsort(centers[peaks])]
    peak_values = centers[peaks]
    peak_counts = counts[peaks]

    if debug:
        plt.figure(figsize=(6, 3))
        plt.plot(centers, counts)
        plt.scatter(peak_values, peak_counts, color="crimson")
        plt.show()

    return peak_values, peak_counts


# Shift PFM phase data so the lower histogram peak (down domains) sits near 0 degrees.
def normalize_phase(phase: np.ndarray) -> np.ndarray:
    """Shift PFM phase data so the lower histogram peak sits near 0.

    Splits the phase histogram at the median, then subtracts the median
    of the lower half. This positions the lower domain population at ~0
    regardless of tails or noise, while preserving the original peak
    separation and data shape.

    When standard PFM phase data covers 0–180° (or 0–270°) for up/down
    domains, the shifted colorbar shows a clean 0-to-upward range via
    the existing percentile or std-based clim logic.
    """
    phase = np.asarray(phase, dtype=float)
    valid = phase[np.isfinite(phase)]
    if valid.size == 0:
        return phase

    median = float(np.nanmedian(valid))
    below = valid[valid < median]
    if below.size == 0:
        return phase

    shift = float(np.nanmedian(below))
    return phase - shift

