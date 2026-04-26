# AFM-tools Usage Guide

`AFM-tools` owns AFM/PFM file loading, channel metadata, AFM-specific unit
handling, and AFM/PFM visualization. Generic figure layout and scale-bar
primitives come from `sci-viz-utils`, but AFM plotting choices stay here.

## Install For Development

```bash
git clone https://github.com/yig319/AFM-tools.git
cd AFM-tools
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

In the local `Pypi_Packages` workspace, `requirements-dev.txt` installs
`../sci-viz-utils` editably first so changes to shared plotting helpers are
visible immediately.

## Load One IBW File

```python
from afm_tools.afm_utils import load_ibw, get_channel

image = load_ibw("sample.ibw")
print(image.sample)
print(image.labels)
height = get_channel(image, label_contains="Height")
```

## Single-Channel Visualization

```python
from afm_tools.afm_viz import AFMVisualizer, prepare_afm_channel_display

channel = get_channel(image, label_contains="Height")
display, metric_text, unit = prepare_afm_channel_display("Height", channel)

visualizer = AFMVisualizer(
    colorbar_setting={
        "colorbar_type": "percent",
        "colorbar_range": (0.2, 99.8),
        "outliers_std": 5,
        "symmetric_clim": False,
        "visible": True,
    },
    zero_mean=False,
    scalebar=True,
)
fig, ax = visualizer.viz(display, scan_size=image.scan_size_m, title="Height", cbar_unit=unit)
```

## Multi-Channel Preview

```python
from afm_tools.afm_viz import plot_afm_channels

fig, axes = plot_afm_channels(
    image,
    selected_channel_indices=[0, 1, 2, 3, 4, 5],
    n_cols=3,
    show_metric_overlay=True,
)
```

## What Belongs Here

Keep AFM/PFM-specific behavior in `AFM-tools`: IBW parsing, channel order,
scan-size interpretation, AFM/PFM units, RMS overlays, and AFM preview style.
Put only generic plotting foundations in `sci-viz-utils`.
