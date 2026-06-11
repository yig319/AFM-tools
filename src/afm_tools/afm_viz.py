"""Visualization helpers for AFM/PFM images."""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sci_viz_utils.figures import make_figure_grid, scalebar as add_scalebar

from .afm_utils import MICRON_UNIT, convert_scan_setting, convert_with_unit, define_percentage_threshold, format_func
from .domain_analysis import normalize_phase

# ── Module-level constants ────────────────────────────────────────────────

_LENGTH_UNIT_SCALES: dict[str, float] = {
    "fm": 1e15,
    "pm": 1e12,
    "nm": 1e9,
    MICRON_UNIT: 1e6,
    "um": 1e6,
    "mm": 1e3,
}

_PREFERRED_CHANNEL_LABELS: tuple[str, ...] = ("Height", "ZSensor", "Amplitude", "Phase")

_DEFAULT_FIGSIZE_SINGLE: tuple[float, float] = (5.0, 4.0)
_DEFAULT_FIGSIZE_SINGLE_WIDE: tuple[float, float] = (6.4, 4.8)
_DEFAULT_FIGSIZE_PER_CHANNEL: tuple[float, float] = (4.2, 3.8)
_DEFAULT_FIGSIZE_PER_IMAGE: tuple[float, float] = (5.0, 4.5)
_DEFAULT_FIGSIZE_SCATTER: tuple[float, float] = (6.0, 4.0)

_MULTI_CHANNEL_ADJUST: dict[str, float] = dict(
    left=0.035, right=0.985, bottom=0.05, top=0.90, wspace=0.02, hspace=0.10,
)
_SINGLE_CHANNEL_ADJUST: dict[str, float] = dict(
    left=0.06, right=0.94, bottom=0.06, top=0.90,
)


class AFMVisualizer:
    """Configurable AFM/PFM single-channel visualizer.

    The call signature follows the usage in PLD_workflow: create one visualizer
    and call ``viz(..., fig=figure, ax=axis, cbar_unit="nm")`` for each channel.

    Colorbar styling is controlled through ``colorbar_setting``. The default
    style is the compact legacy AFM/PFM look: a narrow right-side colorbar,
    inward ticks, small tick labels, and the unit printed above the bar. Set
    ``{"style": "matplotlib"}`` to use Matplotlib's standard colorbar with
    the unit as the side label. Useful compact-style keys include ``size``,
    ``pad``, ``tick_direction``, ``tick_labelsize``, ``tick_length``,
    ``tick_pad``, ``unit_position`` (``"top"`` or ``"side"``),
    ``unit_fontsize``, ``unit_pad``, ``tick_unit``, ``outlier_method``, and
    ``scale_image``. Scale-bar placement is controlled through
    ``scalebar_setting``; increase ``text_offset`` to move the scale label
    farther from the bar.
    """

    # Initialize the visualizer with colorbar, colormap, zero-mean, scalebar, and debug settings.
    def __init__(
        self,
        colorbar_setting: dict | None = None,
        cmap: str = "viridis",
        zero_mean: bool = False,
        scalebar: bool = True,
        scalebar_setting: dict | None = None,
        debug: bool = False,
    ):
        self.colorbar_setting = default_preview_colorbar_setting(colorbar_setting)
        self.cmap = cmap
        self.zero_mean = zero_mean
        self.scalebar = scalebar
        self.scalebar_setting = default_preview_scalebar_setting(scalebar_setting)
        self.debug = debug

    # Look up a key from the merged colorbar settings dict, returning *default* if missing.
    def _setting(self, key, default=None):
        return self.colorbar_setting.get(key, default)

# (static) Filter outlier pixels from a 1D finite-values array using MAD or std.
    @staticmethod
    def _filter_outliers(values: np.ndarray, *, outliers_std: float | None, method: str = "mad") -> np.ndarray:
        """Filter outlier pixels from a 1D finite-values array."""
        if outliers_std is None:
            return values
        if method == "mad":
            center = np.nanmedian(values)
            spread = np.nanmedian(np.abs(values - center))
            if np.isfinite(spread) and spread > 0:
                spread *= 1.4826
            else:
                spread = np.nanstd(values)
        else:
            center = np.nanmean(values)
            spread = np.nanstd(values)
        if np.isfinite(spread) and spread > 0:
            clipped = values[np.abs(values - center) <= outliers_std * spread]
            if clipped.size:
                return clipped
        return values

    # Compute the (vmin, vmax) color limits for an image using configured percentile or range settings.
    def _clim(self, image):
        setting = self.colorbar_setting
        values = np.asarray(image, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None

        finite = self._filter_outliers(
            finite,
            outliers_std=setting.get("outliers_std"),
            method=setting.get("outlier_method", "mad"),
        )

        if setting.get("colorbar_type") == "percent":
            clim = define_percentage_threshold(finite, setting.get("colorbar_range", (2, 98)))
        else:
            clim = setting.get("colorbar_range")

        if clim is None:
            return None
        if setting.get("symmetric_clim", False):
            bound = max(abs(float(clim[0])), abs(float(clim[1])))
            return -bound, bound
        return clim

    # Normalize scan-size input into a dict with image_size, scale_size, units, and pixel_size.
    def _scan_setting(self, scan_size, image):
        if scan_size is None:
            return None
        setting = convert_scan_setting(scan_size)
        setting = dict(setting)
        setting["pixel_size"] = image.shape[1]
        if "image_size" not in setting or setting["image_size"] is None:
            setting["image_size"] = image.shape[1]
        return setting

    # Scale image data from metres into the display unit (nm, µm, mm) when enabled.
    def _scale_for_unit(self, image, cbar_unit: str | None):
        if cbar_unit is None:
            return image
        if not self._setting("scale_image", True):
            return image
        scale = _LENGTH_UNIT_SCALES.get(cbar_unit)
        if scale is None:
            return image

        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return image
        max_abs = np.nanmax(np.abs(finite))
        if np.isfinite(max_abs) and max_abs < 1e-3:
            return image * scale
        return image

    # Dispatcher: route to compact or matplotlib colorbar based on style setting.
    def _add_colorbar(self, fig, ax, im, unit: str | None):
        """Add a colorbar using configurable compact or Matplotlib styling."""
        if not self._setting("visible", True):
            return None

        unit = unit or ""
        style = self._setting("style", "compact")

        if style == "matplotlib":
            return self._add_colorbar_matplotlib(fig, ax, im, unit)
        if style == "compact":
            return self._add_colorbar_compact(fig, ax, im, unit)
        raise ValueError("colorbar_setting['style'] must be 'compact' or 'matplotlib'")

    # Standard matplotlib colorbar with optional unit label on the side.
    def _add_colorbar_matplotlib(self, fig, ax, im, unit: str):
        """Standard Matplotlib colorbar with optional unit label."""
        colorbar = fig.colorbar(
            im,
            ax=ax,
            fraction=self._setting("fraction", 0.046),
            pad=self._setting("pad", 0.04),
        )
        if unit:
            colorbar.set_label(unit)
        return colorbar

    # Compact AFM-style colorbar: narrow right-side, inward ticks, unit above the bar.
    def _add_colorbar_compact(self, fig, ax, im, unit: str):
        """Compact AFM/PFM colorbar with inward ticks and unit above the bar."""
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size=self._setting("size", "5%"),
            pad=self._setting("pad", 0.05),
        )
        do_tick_unit = self._setting("tick_unit", False)
        tick_label_unit = unit if not do_tick_unit else ""
        formatter = plt.FuncFormatter(lambda value, _tick_number: format_func(value, unit=tick_label_unit))
        colorbar = fig.colorbar(im, cax=cax, format=formatter)
        colorbar.solids.set_zorder(0)
        colorbar.ax.yaxis.set_tick_params(
            pad=self._setting("tick_pad", 1),
            labelsize=self._setting("tick_labelsize", 7),
            direction=self._setting("tick_direction", "in"),
            length=self._setting("tick_length", 2),
        )
        colorbar.ax.yaxis.set_tick_params(which="minor", length=0)

        if do_tick_unit and unit:
            unit_position = self._setting("unit_position", "top")
            if unit_position == "top":
                cax.set_title(
                    unit,
                    loc=self._setting("unit_loc", "center"),
                    pad=self._setting("unit_pad", 1),
                    fontsize=self._setting("unit_fontsize", 7),
                )
            elif unit_position == "side":
                colorbar.set_label(unit)
        return colorbar

    # Draw one AFM/PFM channel image with colorbar and optional scale bar. Returns (fig, ax).
    def viz(
        self,
        img,
        scan_size=None,
        fig=None,
        ax=None,
        title: str | None = None,
        cmap: str | None = None,
        cbar_unit: str | None = None,
    ):
        """Plot one AFM/PFM channel and return ``(fig, ax)``.

        Parameters are intentionally plain so notebooks can expose them as
        editable variables: pass the 2D channel in ``img``, the scan size for
        the scale bar, an existing ``fig``/``ax`` if you want layout control,
        and per-plot display choices such as ``title``, ``cmap``, and
        ``cbar_unit``. Height-like data stored in meters is scaled for colorbar
        units of ``"nm"``, ``"um"``, or ``"mm"``.
        """
        image = np.asarray(img, dtype=float)
        image = self._scale_for_unit(image, cbar_unit)
        if self.zero_mean:
            image = image - np.nanmean(image)

        if ax is None:
            if fig is None:
                fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_SINGLE)
            else:
                ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        im = ax.imshow(image, cmap=cmap or self.cmap)
        clim = self._clim(image)
        if clim is not None:
            im.set_clim(*clim)
        ax.set_axis_off()
        if title:
            ax.set_title(title)

        self._add_colorbar(fig, ax, im, cbar_unit)

        if self.scalebar and scan_size is not None:
            self._add_scalebar(ax, scan_size, image)
        return fig, ax

    # Draw a physical scale bar on the axis using the visualizer's scalebar settings.
    def _add_scalebar(self, ax, scan_size, image) -> None:
        """Draw a scale bar on the axis using the visualizer's scalebar settings."""
        setting = self._scan_setting(scan_size, image)
        scalebar_setting = default_preview_scalebar_setting(self.scalebar_setting)
        add_scalebar(
            ax,
            image_size=setting["image_size"],
            scale_size=setting["scale_size"],
            units=setting.get("units", ""),
            pixel_size=setting.get("pixel_size", image.shape[1]),
            **scalebar_setting,
        )


@dataclass(slots=True)
class AfmDataset:
    """AFM/PFM image stack plus metadata needed for repeatable rendering.

    Use :func:`load_afm_dataset` when you want the legacy tuple-style
    ``parse_ibw`` output in a named, easier-to-read container.
    """

    file_path: str
    images: np.ndarray
    sample_name: str
    labels: list[str]
    scan_size: object


@dataclass(slots=True)
class AfmPreviewOptions:
    """Options for :func:`render_afm_preview`.

    ``selected_channel_indices`` controls which channels appear. Passing an
    empty list chooses a sensible default channel from the labels.
    ``colorbar_setting`` is merged with :func:`default_preview_colorbar_setting`,
    so the compact AFM colorbar style remains the default while callers can
    override individual style choices, for example ``{"style": "matplotlib"}``.
    """

    selected_channel_indices: list[int]
    show_metric_overlay: bool = False
    colorbar_setting: dict | None = None
    scalebar_setting: dict | None = None
    cmap: str = "viridis"


@dataclass(slots=True)
class AfmPreviewRender:
    """Rendered AFM/PFM preview and a short status message."""

    figure: object
    message: str


# Load one .ibw file and return it as an AfmDataset named container.
def load_afm_dataset(file_path: str) -> AfmDataset:
    """Load one IBW file with AFM-tools and return a preview-ready dataset."""
    from afm_tools.afm_utils import parse_ibw

    images, sample_name, labels, scan_size = parse_ibw(file_path)
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[2] == 0:
        raise RuntimeError("AFM-tools parsed the file, but no image channels were returned.")

    return AfmDataset(
        file_path=str(file_path),
        images=images,
        sample_name=str(sample_name),
        labels=[str(label) for label in labels],
        scan_size=scan_size,
    )


# Choose the best default preview channel from common AFM/PFM channel names.
def preferred_channel_index(labels: list[str]) -> int:
    """Choose a default preview channel from common AFM/PFM channel names."""
    preferred_labels = _PREFERRED_CHANNEL_LABELS
    for preferred_label in preferred_labels:
        if preferred_label in labels:
            return labels.index(preferred_label)
    return 0


# Return the default compact colorbar style dict, merged with optional user overrides.
def default_preview_colorbar_setting(overrides: dict | None = None) -> dict:
    """Return the default AFM/PFM preview colorbar style — **single source of truth**.

    The default is the compact style used by the original PLD AFM/PFM
    previews: percentile color limits, inward ticks, small labels, and the unit
    above the colorbar. Pass ``{"style": "matplotlib"}`` to switch to
    Matplotlib's standard side-label colorbar, or pass individual compact keys
    such as ``tick_labelsize``, ``tick_pad``, ``unit_position``, ``size``, and
    ``pad`` to tune the restored style. Set ``scale_image=False`` and
    ``tick_unit=True`` to keep data in raw meters and scale only tick labels,
    matching the oldest AFM visualizer behavior.

    Key                  | Alias              | Meaning
    ---------------------|--------------------|--------------------------------------------
    ``colorbar_type``    | ``clim_mode``      | ``"percent"`` uses percentile clipping
    ``colorbar_range``   | ``clim_range``     | Percentile range, e.g. ``(2, 98)``
    ``outliers_std``     | ``clim_sigma``     | MAD/STD multiplier for outlier removal
    ``symmetric_clim``   | ``clim_symmetric`` | Center the colour range around 0
    ``normalize_phase``  |                    | Shift Phase lower domain to ~0 degrees
    """

    setting = {
        "colorbar_type": "percent",
        "colorbar_range": (2, 98),
        "outliers_std": 5,
        "outlier_method": "mad",       # "mad" → median-absolute-deviation; otherwise std-based
        "symmetric_clim": False,
        "visible": True,
        "normalize_phase": True,
        "style": "compact",
        "size": "5%",
        "pad": 0.05,
        "tick_direction": "in",
        "tick_labelsize": 7,
        "tick_length": 2,
        "tick_pad": 1,
        "unit_position": "top",
        "unit_fontsize": 7,
        "unit_pad": 1,
        "tick_unit": True,
        "scale_image": True,
    }
    if overrides:
        setting.update(overrides)
    return setting


# Return the default scalebar style dict (white bar, bottom-right, label above), merged with overrides.
def default_preview_scalebar_setting(overrides: dict | None = None) -> dict:
    """Return the default AFM/PFM scale-bar style.

    ``text_offset`` controls the distance between the label and the bar, and
    ``text_position`` can be ``"above"`` or ``"below"``. The default keeps the
    label just above the bar, which avoids overlap near the image bottom.
    """

    setting = {
        "loc": "br",
        "color": "white",
        "linewidth": 0,
        "text_color": None,
        "text_fontsize": 9,
        "text_offset": 0.35,
        "text_position": "above",
    }
    if overrides:
        setting.update(overrides)
    return setting


# High-level entry point: render a single- or multi-channel AFM preview figure from a dataset.
def render_afm_preview(dataset: AfmDataset, options: AfmPreviewOptions) -> AfmPreviewRender:
    """Render an AFM/PFM preview using the same style as PLD_workflow.

    This is the recommended high-level plotting entry point for notebooks and
    GUI preview tools. It keeps channel selection, figure spacing, scale bars,
    colorbar units, and optional RMS overlays in one package-level function so
    downstream projects do not need to copy plotting logic.
    """
    colorbar_setting = default_preview_colorbar_setting(options.colorbar_setting)
    colorbar_setting["scale_image"] = False
    visualizer = AFMVisualizer(
        colorbar_setting=colorbar_setting,
        cmap=options.cmap,
        zero_mean=False,
        scalebar=True,
        scalebar_setting=default_preview_scalebar_setting(options.scalebar_setting),
        debug=False,
    )

    channel_indices = _normalize_selected_indices(options.selected_channel_indices, len(dataset.labels))
    if not channel_indices:
        channel_indices = [preferred_channel_index(dataset.labels)]

    if len(channel_indices) > 1:
        figure = _render_multi_channel_preview(dataset, options, visualizer, channel_indices)
        return AfmPreviewRender(
            figure=figure,
            message=f"AFM preview updated with {len(channel_indices)} selected channels.",
        )

    figure, metric_text, channel_label = _render_single_channel_preview(
        dataset,
        options,
        visualizer,
        channel_indices[0],
    )
    return AfmPreviewRender(
        figure=figure,
        message=f"AFM preview updated for {channel_label} with RMS = {metric_text}.",
    )


# Notebook-friendly: plot selected channels from a loaded AFM/PFM dataset in a grid.
def plot_afm_channels(
    dataset,
    selected_channel_indices: list[int] | None = None,
    *,
    max_channels: int = 6,
    n_cols: int = 3,
    figsize_per_channel: tuple[float, float] = (4.2, 3.8),
    colorbar_setting: dict | None = None,
    cmap: str = "viridis",
    zero_mean: bool = False,
    scalebar: bool = True,
    channel_cbar_units: dict[str, str] | None = None,
    default_cbar_unit: str | None = None,
    show_metric_overlay: bool = False,
    show_sample_title: bool = True,
):
    """Plot selected channels from one loaded AFM/PFM dataset.

    This is the notebook-friendly version of :func:`render_afm_preview`: it
    does not require ``AfmPreviewOptions`` or any PLD_workflow GUI classes.
    Pass the object returned by ``afm_tools.afm_utils.load_ibw`` or
    :func:`load_afm_dataset`, then edit the plain keyword arguments.
    """
    images, labels, scan_size, sample_name = _afm_dataset_parts(dataset)
    channel_count = images.shape[2] if images.ndim == 3 else 1

    if selected_channel_indices is None:
        channel_indices = list(range(min(max_channels, channel_count)))
    else:
        channel_indices = _normalize_selected_indices(selected_channel_indices, channel_count)
    if not channel_indices:
        channel_indices = [preferred_channel_index(labels)]

    n_channels = len(channel_indices)
    n_cols = max(1, min(int(n_cols), n_channels))
    n_rows = int(math.ceil(n_channels / n_cols))
    figure, axes = make_figure_grid(
        n_channels,
        columns=n_cols,
        figsize=(figsize_per_channel[0] * n_cols, figsize_per_channel[1] * n_rows),
    )

    plot_colorbar_setting = default_preview_colorbar_setting(colorbar_setting)
    plot_colorbar_setting["scale_image"] = False

    visualizer = AFMVisualizer(
        colorbar_setting=plot_colorbar_setting,
        zero_mean=zero_mean,
        scalebar=scalebar,
        debug=False,
    )
    visualizer.cmap = cmap

    channel_cbar_units = channel_cbar_units or {}
    do_norm_phase = plot_colorbar_setting.get("normalize_phase", True)

    display_images: list[np.ndarray] = []
    display_labels: list[str] = []
    display_metrics: list[str] = []
    display_units: list[str] = []
    for channel_index in channel_indices:
        channel_label = labels[channel_index] if channel_index < len(labels) else f"channel_{channel_index}"
        image = _afm_channel(images, channel_index)
        display_image, metric_text, inferred_unit = prepare_afm_channel_display(
            channel_label, image, normalize_phase_data=do_norm_phase,
        )
        cbar_unit = channel_cbar_units.get(channel_label, default_cbar_unit or inferred_unit)
        if cbar_unit != inferred_unit:
            display_image = scale_afm_channel_for_unit(channel_label, image, cbar_unit)
        display_images.append(display_image)
        display_labels.append(f"{channel_index}: {channel_label}")
        display_metrics.append(metric_text)
        display_units.append(cbar_unit)

    _render_channel_grid(
        visualizer,
        display_images,
        display_labels,
        display_metrics,
        scan_size,
        figure=figure,
        axes=axes,
        cbar_units=display_units,
        show_metric_overlay=show_metric_overlay,
        sample_name=sample_name,
        show_sample_title=show_sample_title,
    )
    # Reshape flat axes back to 2D for backward compat with old ``squeeze=False`` return.
    axes = axes.reshape(n_rows, n_cols) if n_channels > 1 else axes
    return figure, axes


# Unpack images, labels, scan_size and sample_name from an AFMImage, AfmDataset, or raw array.
def _afm_dataset_parts(dataset):
    """Return ``images, labels, scan_size, sample_name`` from known AFM containers."""
    # Handle raw numpy arrays (2D or 3D) — no metadata, just images
    if isinstance(dataset, np.ndarray):
        images = dataset
        if images.ndim == 2:
            images = images[:, :, np.newaxis]
        if images.ndim != 3 or images.shape[2] == 0:
            raise ValueError("raw array must be a 2D image or a 3D channel stack")
        labels = [f"channel_{index}" for index in range(images.shape[2])]
        return np.asarray(images, dtype=float), labels, None, "Raw array"

    images = getattr(dataset, "images", None)
    if images is None:
        images = getattr(dataset, "data", None)
    if images is None:
        raise TypeError("dataset must have an 'images' or 'data' array, or be a raw numpy array")

    images = np.asarray(images)
    if images.ndim == 2:
        images = images[:, :, np.newaxis]
    if images.ndim != 3 or images.shape[2] == 0:
        raise ValueError("dataset images must be a 2D image or a 3D channel stack")

    labels = getattr(dataset, "labels", None) or [f"channel_{index}" for index in range(images.shape[2])]
    labels = [str(label) for label in labels]
    if len(labels) < images.shape[2]:
        labels.extend(f"channel_{index}" for index in range(len(labels), images.shape[2]))

    scan_size = getattr(dataset, "scan_size", None)
    if scan_size is None:
        scan_size = getattr(dataset, "scan_size_m", None)

    sample_name = getattr(dataset, "sample_name", None)
    if sample_name is None:
        sample_name = getattr(dataset, "sample", None)
    if sample_name is None:
        file_path = getattr(dataset, "file_path", None) or getattr(dataset, "path", None)
        sample_name = Path(file_path).stem if file_path is not None else "AFM/PFM dataset"

    return images, labels, scan_size, str(sample_name)


# Extract one 2D channel slice from a 3D image stack by channel index.
def _afm_channel(images: np.ndarray, channel_index: int) -> np.ndarray:
    """Return one 2D channel from a normalized AFM image stack."""
    return np.asarray(images[:, :, channel_index], dtype=float)


# Shared rendering loop: draw pre-scaled images into a pre-created figure grid with overlays and suptitle.
def _render_channel_grid(
    visualizer: AFMVisualizer,
    images: list[np.ndarray],
    channel_labels: list[str],
    metric_texts: list[str],
    scan_size,
    *,
    figure: "plt.Figure",
    axes: np.ndarray,
    cbar_units: list[str] | None = None,
    show_metric_overlay: bool = False,
    sample_name: str = "",
    show_sample_title: bool = True,
    single_channel_layout: bool = False,
) -> None:
    """Render pre-scaled channels into a pre-created figure grid.

    All image scaling and unit inference must happen BEFORE calling this
    helper — it only handles the rendering loop, metric overlays, suptitle,
    and layout adjustments.

    Parameters
    ----------
    visualizer:
        Configured :class:`AFMVisualizer` instance.
    images:
        Pre-scaled 2D numpy arrays, one per channel.
    channel_labels:
        Display labels for each channel.
    metric_texts:
        Pre-computed RMS metric strings, one per channel.
    scan_size:
        Physical scan size forwarded to the visualizer.
    figure:
        Pre-created Matplotlib figure.
    axes:
        Flat array of Matplotlib axes, one per channel (from
        :func:`sci_viz_utils.figures.make_figure_grid`).
    cbar_units:
        Colorbar unit strings, one per channel.  When None, the visualizerʼs
        built-in unit scaling is used instead.
    show_metric_overlay:
        Whether to draw RMS labels.
    sample_name:
        Sample name used for ``suptitle`` when *show_sample_title* is True.
    show_sample_title:
        Whether to add a figure-level suptitle.
    single_channel_layout:
        If True, apply the single-channel ``subplots_adjust`` preset instead
        of tight_layout.
    """
    multiple = len(images) > 1
    cbar_units = cbar_units or [None] * len(images)
    for ax, img, label, metric, cbar_unit in zip(axes, images, channel_labels, metric_texts, cbar_units):
        visualizer.viz(
            img=img,
            scan_size=scan_size,
            fig=figure,
            ax=ax,
            title=label,
            cbar_unit=cbar_unit,
        )
        if show_metric_overlay and should_show_metric_overlay(label, multiple_plots=multiple):
            add_metric_overlay(ax, f"RMS = {metric}")

    if show_sample_title:
        figure.suptitle(sample_name, fontsize=12)

    if single_channel_layout:
        figure.subplots_adjust(**_SINGLE_CHANNEL_ADJUST)
    else:
        figure.set_layout_engine(None)
        figure.tight_layout()
        if show_sample_title:
            figure.subplots_adjust(top=0.90)


# Render a multi-channel preview grid (2-3 columns depending on channel count).
def _render_multi_channel_preview(dataset, options, visualizer, channel_indices):
    n_channels = len(channel_indices)
    n_cols = 2 if n_channels <= 4 else 3
    figure, axes = make_figure_grid(
        n_channels,
        columns=n_cols,
        figsize=(_DEFAULT_FIGSIZE_PER_CHANNEL[0] * n_cols,
                 _DEFAULT_FIGSIZE_PER_CHANNEL[1] * int(math.ceil(n_channels / n_cols))),
    )

    display_images: list[np.ndarray] = []
    channel_labels: list[str] = []
    metric_texts: list[str] = []
    cbar_units: list[str] = []
    for channel_index in channel_indices:
        channel_label = dataset.labels[channel_index]
        image = np.asarray(dataset.images[:, :, channel_index], dtype=float)
        display_image, metric_text, unit = prepare_afm_channel_display(channel_label, image)
        display_images.append(display_image)
        channel_labels.append(channel_label)
        metric_texts.append(metric_text)
        cbar_units.append(unit)

    _render_channel_grid(
        visualizer,
        display_images,
        channel_labels,
        metric_texts,
        dataset.scan_size,
        figure=figure,
        axes=axes,
        cbar_units=cbar_units,
        show_metric_overlay=options.show_metric_overlay,
        sample_name=dataset.sample_name,
        show_sample_title=True,
    )
    figure.subplots_adjust(**_MULTI_CHANNEL_ADJUST)
    return figure


# Render a single-channel preview figure (one large panel with suptitle).
def _render_single_channel_preview(dataset, options, visualizer, channel_index):
    channel_label = dataset.labels[channel_index]
    image = np.asarray(dataset.images[:, :, channel_index], dtype=float)
    display_image, metric_text, unit = prepare_afm_channel_display(channel_label, image)

    figure, axis = plt.subplots(figsize=_DEFAULT_FIGSIZE_SINGLE_WIDE)
    axes = np.array([axis])

    _render_channel_grid(
        visualizer,
        [display_image],
        [channel_label],
        [metric_text],
        dataset.scan_size,
        figure=figure,
        axes=axes,
        cbar_units=[unit],
        show_metric_overlay=options.show_metric_overlay,
        sample_name=f"{dataset.sample_name} - {channel_label}",
        show_sample_title=True,
        single_channel_layout=True,
    )
    return figure, metric_text, channel_label


# Draw a small RMS roughness label (white box, black text) in the top-left corner of an axis.
def add_metric_overlay(axis, metric_text: str) -> None:
    """Draw a small metric label directly on top of one preview image."""
    axis.text(
        0.02,
        0.98,
        metric_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3.0},
    )


# Compute root-mean-square (RMS) of finite pixels after subtracting the mean.
def compute_rms_metric(image: np.ndarray) -> float:
    """Compute RMS value over finite pixels after subtracting the mean."""
    values = np.asarray(image, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    centered = finite_values - float(finite_values.mean())
    return float(np.sqrt(np.mean(centered**2)))


# Return formatted RMS text and display unit — backward-compat wrapper (accepts raw or pre-scaled).
def describe_afm_metric(channel_label: str, image: np.ndarray) -> tuple[str, str]:
    """Return formatted RMS text and the colorbar unit for one channel.

    .. deprecated:: (next)
        The two-argument ``(channel_label, image)`` form will be removed in a
        future version.  Pre-scale the image with :func:`scale_afm_channel_for_display`
        and pass ``(display_image, unit)`` instead.  For now both forms work.
    """
    if isinstance(channel_label, str):
        display_image, unit = scale_afm_channel_for_display(channel_label, image)
    else:
        display_image = np.asarray(channel_label, dtype=float)
        unit = str(image or "")
    return _describe_afm_metric(display_image, unit)


# Format RMS metric text from an already-scaled display image (internal fast path).
def _describe_afm_metric(display_image: np.ndarray, unit: str) -> tuple[str, str]:
    """Format RMS metric text from an already-scaled display image."""
    rms_value = compute_rms_metric(display_image)
    if unit == "deg":
        return f"{rms_value:.2f} deg", "deg"
    return f"{rms_value:.3g} {unit}", unit


# Scale one raw channel into display units and return (display_image, metric_text, unit).
def prepare_afm_channel_display(
    channel_label: str,
    image: np.ndarray,
    *,
    normalize_phase_data: bool = True,
) -> tuple[np.ndarray, str, str]:
    """Return display image, metric text, and colorbar unit for one channel."""
    display_image, unit = scale_afm_channel_for_display(channel_label, image, normalize_phase_data=normalize_phase_data)
    metric_text, _ = _describe_afm_metric(display_image, unit)
    return display_image, metric_text, unit


# Infer the display unit for a channel, optionally normalize phase, then scale to that unit.
def scale_afm_channel_for_display(
    channel_label: str,
    image: np.ndarray,
    *,
    normalize_phase_data: bool = True,
) -> tuple[np.ndarray, str]:
    """Scale one raw AFM/PFM channel into its default display unit.

    When *normalize_phase_data* is True (default) and the channel is
    identified as a Phase channel, the data is shifted so its lower
    histogram peak sits near 0 degrees (see
    :func:`~domain_analysis.normalize_phase`). Pass
    ``normalize_phase_data=False`` to keep the raw phase offset.
    """
    unit = infer_afm_channel_unit(channel_label, image)
    transformed = np.asarray(image, dtype=float)
    if unit == "deg" and normalize_phase_data:
        transformed = normalize_phase(transformed)
    return scale_afm_channel_for_unit(channel_label, transformed, unit), unit


# Convert raw AFM channel data (in metres) to the specified display unit (nm, µm, deg, etc.).
def scale_afm_channel_for_unit(channel_label: str, image: np.ndarray, unit: str) -> np.ndarray:
    """Scale one raw AFM/PFM channel for display in ``unit``."""
    if unit == "deg":
        return np.asarray(image, dtype=float)

    values_m = _length_values_in_meters(channel_label, image)
    scale = _LENGTH_UNIT_SCALES.get(unit)
    return values_m * scale if scale is not None else np.asarray(image, dtype=float)


# Guess the best display unit (nm, µm, deg, etc.) for a channel from its label and data values.
def infer_afm_channel_unit(channel_label: str, image: np.ndarray, *, metric_value: float | None = None) -> str:
    """Infer a display unit for an AFM/PFM channel.

    Phase-like channels use degrees. Other channels are treated as length-like
    data and choose an engineering length unit from the actual channel values,
    avoiding a fallback to raw meters or a fixed unit per channel label.
    """

    normalized = channel_label.strip().lower().replace(" ", "")
    if "phase" in normalized:
        return "deg"

    if metric_value is not None and np.isfinite(metric_value):
        values_m = _length_values_in_meters(channel_label, np.asarray([metric_value], dtype=float))
    else:
        values_m = _length_values_in_meters(channel_label, image)
    finite = values_m[np.isfinite(values_m)]
    reference = float(np.nanmax(np.abs(finite))) if finite.size else 0.0
    if not np.isfinite(reference) or reference == 0:
        return "nm"

    unit_text = convert_with_unit(reference)
    return unit_text.split()[-1] if " " in unit_text else "nm"


# Normalize length-like channel values to metres, handling LatAmplitude special case.
def _length_values_in_meters(channel_label: str, image: np.ndarray) -> np.ndarray:
    """Return length-like channel values normalized to meters."""
    values = np.asarray(image, dtype=float)
    normalized = channel_label.strip().lower().replace(" ", "")
    if normalized == "latamplitude":
        return values * _lat_amplitude_meter_scale(values)
    return values


# Infer the metres scale factor for LatAmplitude channels (Asylum stores these in µm or m).
def _lat_amplitude_meter_scale(values: np.ndarray) -> float:
    """Infer the raw LatAmplitude length scale.

    Asylum lateral-amplitude channels commonly arrive in micrometers, while
    very small exported values are already in meters. Normalize both cases to
    meters before choosing the display unit.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    max_abs = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(max_abs) or max_abs == 0:
        return 1.0
    if max_abs < 1e-6:
        return 1.0
    if max_abs < 1:
        return 1e-6
    return 1e-9


# Decide whether to draw an RMS overlay: always for single plots; for grids, only on Height.
def should_show_metric_overlay(channel_label: str, *, multiple_plots: bool) -> bool:
    """Return whether the RMS label should be drawn for one channel."""
    if not multiple_plots:
        return True
    return channel_label.strip().lower() == "height"


# Return selected channel indices as an in-range, deduplicated list of ints.
def _normalize_selected_indices(selected_channel_indices: list[int], channel_count: int) -> list[int]:
    """Return selected channel indices as an in-range, deduplicated list."""
    normalized: list[int] = []
    seen: set[int] = set()
    for index in selected_channel_indices:
        if index < 0 or index >= channel_count:
            continue
        if index in seen:
            continue
        normalized.append(int(index))
        seen.add(int(index))
    return normalized


# Scatter-plot RMS roughness vs a growth parameter, with per-sample means as red lines.
def plot_roughness_comparison(
    df: "pd.DataFrame",
    target_param: str,
    *,
    ax: "plt.Axes | None" = None,
    title: str | None = None,
    roughness_col: str = "rms_nm",
) -> "tuple[plt.Figure, plt.Axes]":
    """Scatter-plot roughness vs. a growth parameter, one point per scan.

    Each sample's individual scans are shown as scatter points; the sample
    mean is drawn as a red line with a label.

    Parameters
    ----------
    df : DataFrame
        Must contain ``sample``, *roughness_col*, and *target_param* columns.
    target_param : str
        Column name for the x-axis (e.g. ``"temperature_C"``).
    ax : Axes, optional
        Matplotlib axis to plot on.  Created if omitted.
    title : str, optional
        Plot title.
    roughness_col : str
        Column name for the y-axis roughness values (default ``"rms_nm"``).
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_SCATTER)
    else:
        fig = ax.figure

    param_by_sample = df.groupby("sample")[target_param].first()
    norm = plt.Normalize(param_by_sample.min(), param_by_sample.max())
    cmap = plt.cm.viridis

    for sample in sorted(df["sample"].unique()):
        subset = df[df["sample"] == sample]
        val = param_by_sample[sample]
        ax.scatter([val] * len(subset), subset[roughness_col],
                   color=cmap(norm(val)), alpha=0.6, s=50, zorder=3)
        mean_val = subset[roughness_col].mean()
        ax.plot([val - 0.02, val + 0.02], [mean_val, mean_val],
                "r-", linewidth=2, zorder=4)
        ax.annotate(sample, (val, mean_val), fontsize=7,
                    ha="center", va="bottom", color="red")

    ax.set_xlabel(target_param)
    ax.set_ylabel(f"RMS Roughness ({roughness_col})")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    return fig, ax


# Simple scatter plot from one or two DataFrames with optional per-point labels.
def df_scatter(
    df1,
    df2=None,
    xaxis=None,
    yaxis=None,
    label_with=None,
    style: str = "simple",
    ax=None,
    **kwargs,
):
    """Draw a simple scatter plot from one or two data frames."""
    if xaxis is None or yaxis is None:
        raise ValueError("xaxis and yaxis are required")
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", _DEFAULT_FIGSIZE_SINGLE))
    else:
        fig = ax.figure

    ax.scatter(df1[xaxis], df1[yaxis], label=kwargs.pop("label1", None), **kwargs)
    if df2 is not None:
        ax.scatter(df2[xaxis], df2[yaxis], label=kwargs.pop("label2", None), marker="s")
    if label_with and style == "simple":
        for _, row in df1.iterrows():
            ax.annotate(str(row[label_with]), (row[xaxis], row[yaxis]), fontsize=8)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    if df2 is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


# Plot the Height channel of multiple AFM images in a grid with colour bars and scale bars.
def show_topography_grid(
    images,
    *,
    labels: list[str] | None = None,
    channel: str = "Height",
    max_cols: int = 3,
    figsize_per_image: tuple[float, float] = (5, 4.5),
    cmap: str = "viridis",
    cbar_unit: str = "nm",
    colorbar_setting: dict | None = None,
    show_metric_overlay: bool = True,
    **viz_kwargs,
) -> "tuple[plt.Figure, np.ndarray]":
    """Plot the Height channel of multiple AFM images in a grid.

    Each image gets its own panel with a compact colour bar, scale bar,
    and unit scaling via :class:`AFMVisualizer`.

    Parameters
    ----------
    images : list of AFMImage or list of Path/str
        AFM images or .ibw file paths to display.
    labels : list of str, optional
        Per-image titles (defaults to the sample name from each image).
    channel : str
        Channel label to extract (default ``"Height"``).
    max_cols : int
        Maximum number of columns in the grid.
    figsize_per_image : tuple
        Figure size per image panel ``(width, height)``.
    cmap : str
        Colormap name.
    cbar_unit : str
        Colour bar unit (e.g. ``"nm"``, ``"µm"``).
    colorbar_setting : dict, optional
        Overrides for :func:`default_preview_colorbar_setting`.
    show_metric_overlay : bool
        If True, draw an RMS roughness label on each panel.
    **viz_kwargs
        Extra keyword arguments forwarded to :meth:`AFMVisualizer.viz`.

    Returns
    -------
    (fig, axes)
    """
    from .afm_utils import get_channel, load_ibw

    # Accept paths or already-loaded AFMImage objects
    loaded: list = []
    for img in images:
        if isinstance(img, (str, Path)):
            loaded.append(load_ibw(str(img)))
        else:
            loaded.append(img)

    n = len(loaded)
    ncols = min(n, max_cols)
    fig, axes = make_figure_grid(
        n,
        columns=ncols,
        figsize=(figsize_per_image[0] * ncols, figsize_per_image[1] * int(math.ceil(n / ncols))),
        layout=None,
    )

    cbar = default_preview_colorbar_setting(colorbar_setting)
    cbar["scale_image"] = False
    if "size" not in (colorbar_setting or {}):
        cbar["size"] = "4%"
    viz = AFMVisualizer(colorbar_setting=cbar, cmap=cmap)

    for i, (ax, img) in enumerate(zip(axes, loaded)):
        height = get_channel(img, label_contains=channel)
        if height is None:
            height = get_channel(img, index=0)
        display_image, unit = scale_afm_channel_for_display(channel, height)
        title = labels[i] if labels and i < len(labels) else img.sample
        viz.viz(display_image, scan_size=img.scan_size_m, fig=fig, ax=ax,
                title=title, cbar_unit=cbar_unit, cmap=cmap, **viz_kwargs)
        if show_metric_overlay:
            metric_text, _ = describe_afm_metric(display_image, unit)
            add_metric_overlay(ax, f"RMS = {metric_text}")

    fig.tight_layout()
    return fig, axes


class TipPositionAnalyzer:
    """Simple frequency-vs-amplitude tune plot for tip-position analysis."""

    # Plot frequency vs amplitude for tip-position tune analysis, with optional per-curve labels.
    def show_tune(self, freq, amps, colors=None, positions=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_SCATTER)
        else:
            fig = ax.figure

        freq = np.asarray(freq)
        colors = colors or [None] * len(amps)
        positions = positions or [f"position {idx}" for idx in range(len(amps))]
        for amp, color, label in zip(amps, colors, positions):
            ax.plot(freq, np.asarray(amp), color=color, label=label)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax


class tip_potisition_analyzer(TipPositionAnalyzer):
    """Legacy spelling kept for backward compatibility with existing notebooks.

    .. deprecated::
        Use :class:`TipPositionAnalyzer` instead.
    """


__all__ = [
    "AfmDataset",
    "AfmPreviewOptions",
    "AfmPreviewRender",
    "AFMVisualizer",
    "TipPositionAnalyzer",
    "add_metric_overlay",
    "compute_rms_metric",
    "convert_with_unit",
    "default_preview_colorbar_setting",
    "default_preview_scalebar_setting",
    "describe_afm_metric",
    "df_scatter",
    "infer_afm_channel_unit",
    "load_afm_dataset",
    "normalize_phase",
    "plot_afm_channels",
    "plot_roughness_comparison",
    "prepare_afm_channel_display",
    "preferred_channel_index",
    "render_afm_preview",
    "scale_afm_channel_for_display",
    "scale_afm_channel_for_unit",
    "should_show_metric_overlay",
    "show_topography_grid",
    "tip_potisition_analyzer",
]
