from __future__ import annotations

import numpy as np

from afm_tools.afm_image_analyzer import domain_fraction


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

