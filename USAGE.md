# AFM-tools Usage Guide

`AFM-tools` is responsible for AFM/PFM file loading, Igor/IBW metadata parsing,
AFM-specific cleaning, roughness/domain metrics, unit handling, and AFM/PFM
visualization. Generic figure mechanics come from `sci-viz-utils`.

## Install For Development

```bash
cd AFM-tools
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Workflow: Load IBW Files And Choose Channels

Main entry point: `load_ibw`.
Helpers: `parse_ibw`, `parse_notes`, `get_channel`, `preferred_channel_index`.

Use `load_ibw(path)` for new code. It returns an `AFMImage` with `data`,
`labels`, `sample`, `scan_size_m`, and parsed metadata. Use `get_channel` by
index when the channel order is known or by `label_contains` when notebooks need
to survive different export label names.

```python
from afm_tools.afm_utils import load_ibw, get_channel

image = load_ibw("sample.ibw")
height = get_channel(image, label_contains="Height")
phase = get_channel(image, label_contains="Phase")
```

Use `parse_ibw` only when an older notebook expects `(images, sample_name,
labels, scan_size_m)`.

## Workflow: Flatten, Clean, And Measure Images

Main entry points: `flatten_plane`, `afm_RMS_roughness`, `domain_fraction`.
Helpers: `fit_background`, `polyfit2d`, `remove_outliers`,
`remove_surface_particles`, `calculate_height_profile`.

Use flattening before roughness when a scan has tilt or curvature. Use outlier
removal for dust/particles only when you intentionally want surface texture
without isolated particles. Inputs are 2D arrays; metric outputs are floats,
profiles, masks, or small summary tables.

```python
from afm_tools.afm_image_analyzer import flatten_plane, remove_outliers, afm_RMS_roughness, domain_fraction

flat = flatten_plane(height, order=1)
clean = remove_outliers(flat, threshold=5)
rms = afm_RMS_roughness(clean)
fraction, mask, threshold = domain_fraction(phase)
```

## Workflow: Render AFM/PFM Preview Figures

Main entry point: `render_afm_preview`.
Helpers: `load_afm_dataset`, `AfmPreviewOptions`, `AFMVisualizer`,
`prepare_afm_channel_display`, `infer_afm_channel_unit`, `get_afm_cmap`,
`add_metric_overlay`.

Use this path for notebook or GUI previews that need consistent colormaps,
units, colorbar limits, scale bars, and RMS overlays.

```python
from afm_tools.afm_viz import load_afm_dataset, render_afm_preview, AfmPreviewOptions

dataset = load_afm_dataset("sample.ibw")
result = render_afm_preview(dataset, AfmPreviewOptions(selected_channel_indices=[]))
result.figure.savefig("afm_preview.png", dpi=300)
```

Common tuning: set `selected_channel_indices` for specific channels, override
`colorbar_setting` for fixed limits, and override `scalebar_setting` when the
scan size metadata is missing or wrong.

## Workflow: Batch Summaries And Simple Plots

Main entry points: `roughness_summary`, `plot_channels`, `show_pfm_images`,
`df_scatter`, `plot_surface_3d`, `find_histogram_peaks`.

Use `roughness_summary(files, channel_index=0, flatten=True)` for many IBW files.
Use the simple plotting helpers for quick checks; use `render_afm_preview` for
publication-style single-scan previews.

```python
from pathlib import Path
from afm_tools.afm_image_analyzer import roughness_summary, plot_channels

files = sorted(Path("data").glob("*.ibw"))
summary = roughness_summary(files, channel_index=0, flatten=True)
fig, axes = plot_channels(image)
```

## Function Map

This compact map is for lookup after you know the workflow you need.

### `afm_tools.afm_image_analyzer`
Functions: `polyfit2d(x, y, z, kx=3, ky=3, order=None)`, `fit_background(img, degrees=(3, 3), viz=False)`, `flatten_plane(image, order=1)`, `remove_surface_particles(img, threshold=3, viz=False)`, `remove_outliers(image, threshold=4.0)`, `afm_RMS_roughness(height)`, `rms_roughness(height)`, `calculate_height_profile(image, p0, p1)`, `line_profile(image, p0, p1)`, `domain_fraction(image, threshold=None)`, `roughness_summary(files, channel_index=0, flatten=True, sample_parser=None)`, `plot_channels(image, max_channels=6, cmap='viridis')`

### `afm_tools.afm_utils`
Functions: `parse_notes(note)`, `load_ibw(path, mode=None, reorder_channels=True)`, `parse_ibw(file, mode=None)`, `get_channel(image, index=0, label_contains=None)`, `define_percentage_threshold(image, percentage=(2, 98))`, `convert_scan_setting(scan_size)`, `flexible_round(value, sig_digits=1)`, `convert_with_unit(value, unit='m')`, `format_func(value, unit='')`
Classes: `AFMImage`

### `afm_tools.afm_viz`
Functions: `load_afm_dataset(file_path)`, `preferred_channel_index(labels)`, `default_preview_colorbar_setting(overrides=None)`, `default_preview_scalebar_setting(overrides=None)`, `render_afm_preview(dataset, options)`, `plot_afm_channels(dataset, selected_channel_indices=None, *, max_channels=6, n_cols=3, figsize_per_channel=(4.2, 3.8), colorbar_setting=None, cmap='viridis', zero_mean=False, scalebar=True, channel_cbar_units=None, default_cbar_unit=None, show_metric_overlay=False, show_sample_title=True)`, `add_metric_overlay(axis, metric_text)`, `compute_rms_metric(image)`, `describe_afm_metric(channel_label, image)`, `prepare_afm_channel_display(channel_label, image)`, `scale_afm_channel_for_display(channel_label, image)`, `scale_afm_channel_for_unit(channel_label, image, unit)`, `infer_afm_channel_unit(channel_label, image, *, metric_value=None)`, `should_show_metric_overlay(channel_label, *, multiple_plots)`, `show_pfm_images(imgs, labels=None, fig_name=None, cmap='viridis', cols=3, **kwargs)`, `df_scatter(df1, df2=None, xaxis=None, yaxis=None, label_with=None, style='simple', ax=None, **kwargs)`
Classes: `AFMVisualizer` (viz), `AfmDataset`, `AfmPreviewOptions`, `AfmPreviewRender`, `tip_potisition_analyzer` (show_tune)

### `afm_tools.cmap`
Functions: `get_afm_cmap(channel_label=None, default='viridis')`

### `afm_tools.domain_analysis`
Functions: `find_histogram_peaks(image, bins=256, num_peaks=2, distance='auto', threshold_factor=1.5, min_prominence=5, debug=False)`

### `afm_tools.drawing_3d`
Functions: `plot_surface_3d(height, scan_size=None, ax=None, cmap='viridis', stride=1)`

### `afm_tools.igor.binarywave`
Functions: `load(filename)`
Classes: `StaticStringField` (post_unpack), `NullStaticStringField`, `DynamicWaveDataField1` (pre_pack, pre_unpack, unpack), `DynamicWaveDataField5`, `DynamicStringField` (pre_unpack), `DynamicWaveNoteField`, `DynamicDependencyFormulaField`, `DynamicDataUnitsField`, `DynamicDimensionUnitsField`, `DynamicLabelsField` (post_unpack), `DynamicStringIndicesDataField` (pre_pack, pre_unpack, post_unpack), `DynamicVersionField` (pre_pack, post_unpack), `DynamicWaveField` (post_unpack)

### `afm_tools.igor.igor`
Functions: `load(filename)`, `save(filename)`
Classes: `StaticStringField` (post_unpack), `NullStaticStringField`, `DynamicWaveDataField1` (pre_pack, pre_unpack, unpack), `DynamicWaveDataField5`, `DynamicStringField` (pre_unpack), `DynamicWaveNoteField`, `DynamicDependencyFormulaField`, `DynamicDataUnitsField`, `DynamicDimensionUnitsField`, `DynamicLabelsField` (post_unpack), `DynamicStringIndicesDataField` (pre_pack, pre_unpack, post_unpack), `DynamicVersionField` (pre_pack, post_unpack), `DynamicWaveField` (post_unpack)
