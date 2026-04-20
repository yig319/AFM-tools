"""Visualization helpers for AFM/PFM images."""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .afm_utils import MICRON_UNIT, convert_scan_setting, convert_with_unit, define_percentage_threshold, format_func
from .viz_layout import layout_fig, scalebar as add_scalebar


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

    def __init__(
        self,
        colorbar_setting: dict | None = None,
        cmap: str = "viridis",
        zero_mean: bool = False,
        scalebar: bool = True,
        scalebar_setting: dict | None = None,
        debug: bool = False,
    ):
        self.colorbar_setting = colorbar_setting or {}
        self.cmap = cmap
        self.zero_mean = zero_mean
        self.scalebar = scalebar
        self.scalebar_setting = scalebar_setting or {}
        self.debug = debug

    def _setting(self, key, default=None):
        return self.colorbar_setting.get(key, default)

    def _clim(self, image):
        setting = self.colorbar_setting
        values = np.asarray(image, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None

        outliers_std = setting.get("outliers_std")
        if outliers_std is not None:
            if setting.get("outlier_method", "mad") == "mad":
                center = np.nanmedian(finite)
                spread = np.nanmedian(np.abs(finite - center))
                if np.isfinite(spread) and spread > 0:
                    spread *= 1.4826
                else:
                    spread = np.nanstd(finite)
            else:
                center = np.nanmean(finite)
                spread = np.nanstd(finite)
            if np.isfinite(spread) and spread > 0:
                clipped = finite[np.abs(finite - center) <= outliers_std * spread]
                if clipped.size:
                    finite = clipped

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

    def _scan_setting(self, scan_size, image):
        if scan_size is None:
            return None
        setting = convert_scan_setting(scan_size)
        setting = dict(setting)
        setting["pixel_size"] = image.shape[1]
        if "image_size" not in setting or setting["image_size"] is None:
            setting["image_size"] = image.shape[1]
        return setting

    def _scale_for_unit(self, image, cbar_unit: str | None):
        if cbar_unit is None:
            return image
        if not self._setting("scale_image", True):
            return image
        unit_scale = {"fm": 1e15, "pm": 1e12, "nm": 1e9, MICRON_UNIT: 1e6, "um": 1e6, "mm": 1e3}
        scale = unit_scale.get(cbar_unit)
        if scale is None:
            return image

        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return image
        max_abs = np.nanmax(np.abs(finite))
        if np.isfinite(max_abs) and max_abs < 1e-3:
            return image * scale
        return image

    def _add_colorbar(self, fig, ax, im, unit: str | None):
        """Add a colorbar using configurable compact or Matplotlib styling."""
        if not self._setting("visible", True):
            return None

        unit = unit or ""
        style = self._setting("style", self._setting("colorbar_style", "compact"))
        unit_position = self._setting("unit_position", "top")

        if style == "matplotlib":
            colorbar = fig.colorbar(
                im,
                ax=ax,
                fraction=self._setting("fraction", 0.046),
                pad=self._setting("pad", 0.04),
            )
            if unit:
                colorbar.set_label(unit)
            return colorbar

        if style != "compact":
            raise ValueError("colorbar_setting['style'] must be 'compact' or 'matplotlib'")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size=self._setting("size", "5%"),
            pad=self._setting("pad", 0.05),
        )
        tick_unit = unit if self._setting("tick_unit", False) else ""
        formatter = plt.FuncFormatter(lambda value, _tick_number: format_func(value, unit=tick_unit))
        colorbar = fig.colorbar(im, cax=cax, format=formatter)
        colorbar.ax.yaxis.set_tick_params(
            pad=self._setting("tick_pad", 1),
            labelsize=self._setting("tick_labelsize", 7),
            direction=self._setting("tick_direction", "in"),
            length=self._setting("tick_length", 2),
        )
        if unit and unit_position == "top":
            cax.set_title(
                unit,
                loc=self._setting("unit_loc", "center"),
                pad=self._setting("unit_pad", 1),
                fontsize=self._setting("unit_fontsize", 7),
            )
        elif unit and unit_position == "side":
            colorbar.set_label(unit)
        return colorbar

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
                fig, ax = plt.subplots(figsize=(5, 4))
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
        return fig, ax


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


def preferred_channel_index(labels: list[str]) -> int:
    """Choose a default preview channel from common AFM/PFM channel names."""
    preferred_labels = ("Height", "ZSensor", "Amplitude", "Phase")
    for preferred_label in preferred_labels:
        if preferred_label in labels:
            return labels.index(preferred_label)
    return 0


def default_preview_colorbar_setting(overrides: dict | None = None) -> dict:
    """Return the default AFM/PFM preview colorbar style.

    The default is the compact style used by the original PLD AFM/PFM
    previews: percentile color limits, inward ticks, small labels, and the unit
    above the colorbar. Pass ``{"style": "matplotlib"}`` to switch to
    Matplotlib's standard side-label colorbar, or pass individual compact keys
    such as ``tick_labelsize``, ``tick_pad``, ``unit_position``, ``size``, and
    ``pad`` to tune the restored style. Set ``scale_image=False`` and
    ``tick_unit=True`` to keep data in raw meters and scale only tick labels,
    matching the oldest AFM visualizer behavior.
    """

    setting = {
        "colorbar_type": "percent",
        "colorbar_range": (0.2, 99.8),
        "outliers_std": 5,
        "outlier_method": "mad",
        "symmetric_clim": False,
        "visible": True,
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
        "tick_unit": False,
        "scale_image": True,
    }
    if overrides:
        setting.update(overrides)
    return setting


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
    n_rows = int(math.ceil(n_channels / float(n_cols)))
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_channel[0] * n_cols, figsize_per_channel[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    plot_colorbar_setting = colorbar_setting or {
            "colorbar_type": "percent",
            "colorbar_range": (2, 98),
            "outliers_std": 5,
            "symmetric_clim": True,
            "visible": True,
        }
    plot_colorbar_setting = dict(plot_colorbar_setting)
    plot_colorbar_setting["scale_image"] = False

    visualizer = AFMVisualizer(
        colorbar_setting=plot_colorbar_setting,
        zero_mean=zero_mean,
        scalebar=scalebar,
        debug=False,
    )

    channel_cbar_units = channel_cbar_units or {}
    for axis, channel_index in zip(axes_flat, channel_indices):
        channel_label = labels[channel_index] if channel_index < len(labels) else f"channel_{channel_index}"
        image = _afm_channel(images, channel_index)
        display_image, metric_text, inferred_unit = prepare_afm_channel_display(channel_label, image)
        cbar_unit = channel_cbar_units.get(channel_label, default_cbar_unit or inferred_unit)
        if cbar_unit != inferred_unit:
            display_image = scale_afm_channel_for_unit(channel_label, image, cbar_unit)
        visualizer.viz(
            img=display_image,
            scan_size=scan_size,
            fig=figure,
            ax=axis,
            title=f"{channel_index}: {channel_label}",
            cmap=cmap,
            cbar_unit=cbar_unit,
        )
        if show_metric_overlay and should_show_metric_overlay(channel_label, multiple_plots=n_channels > 1):
            add_metric_overlay(axis, f"RMS = {metric_text}")

    for axis in axes_flat[n_channels:]:
        axis.set_visible(False)

    if show_sample_title:
        figure.suptitle(sample_name, fontsize=12)
    figure.tight_layout()
    if show_sample_title:
        figure.subplots_adjust(top=0.90)
    return figure, axes


def _afm_dataset_parts(dataset):
    """Return ``images, labels, scan_size, sample_name`` from known AFM containers."""
    images = getattr(dataset, "images", None)
    if images is None:
        images = getattr(dataset, "data", None)
    if images is None:
        raise TypeError("dataset must have an 'images' or 'data' array")

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


def _afm_channel(images: np.ndarray, channel_index: int) -> np.ndarray:
    """Return one 2D channel from a normalized AFM image stack."""
    return np.asarray(images[:, :, channel_index], dtype=float)


def _render_multi_channel_preview(dataset, options, visualizer, channel_indices):
    n_channels = len(channel_indices)
    n_cols = 2 if n_channels <= 4 else 3
    n_rows = int(math.ceil(n_channels / float(n_cols)))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for axis, channel_index in zip(axes_flat, channel_indices):
        channel_label = dataset.labels[channel_index]
        image = np.asarray(dataset.images[:, :, channel_index], dtype=float)
        display_image, metric_text, colorbar_unit = prepare_afm_channel_display(channel_label, image)
        visualizer.viz(
            img=display_image,
            scan_size=dataset.scan_size,
            fig=figure,
            ax=axis,
            title=channel_label,
            cbar_unit=colorbar_unit,
        )
        if options.show_metric_overlay and should_show_metric_overlay(channel_label, multiple_plots=True):
            add_metric_overlay(axis, f"RMS = {metric_text}")

    for axis in axes_flat[n_channels:]:
        axis.set_visible(False)

    figure.suptitle(dataset.sample_name, fontsize=12)
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.05, top=0.90, wspace=0.02, hspace=0.10)
    return figure


def _render_single_channel_preview(dataset, options, visualizer, channel_index):
    channel_label = dataset.labels[channel_index]
    image = np.asarray(dataset.images[:, :, channel_index], dtype=float)
    display_image, metric_text, colorbar_unit = prepare_afm_channel_display(channel_label, image)

    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    visualizer.viz(
        img=display_image,
        scan_size=dataset.scan_size,
        fig=figure,
        ax=axis,
        title=None,
        cbar_unit=colorbar_unit,
    )
    figure.suptitle(f"{dataset.sample_name} - {channel_label}", fontsize=12)
    if options.show_metric_overlay:
        add_metric_overlay(axis, f"RMS = {metric_text}")
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.90)
    return figure, metric_text, channel_label


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


def compute_rms_metric(image: np.ndarray) -> float:
    """Compute RMS value over finite pixels after subtracting the mean."""
    values = np.asarray(image, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    centered = finite_values - float(finite_values.mean())
    return float(np.sqrt(np.mean(centered**2)))


def describe_afm_metric(channel_label: str, image: np.ndarray) -> tuple[str, str]:
    """Return formatted RMS text and the colorbar unit for one channel."""
    display_image, unit = scale_afm_channel_for_display(channel_label, image)
    rms_value = compute_rms_metric(display_image)
    if unit == "deg":
        return f"{rms_value:.2f} deg", "deg"
    return f"{rms_value:.3g} {unit}", unit


def prepare_afm_channel_display(channel_label: str, image: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Return display image, metric text, and colorbar unit for one channel."""
    display_image, unit = scale_afm_channel_for_display(channel_label, image)
    metric_text, _unit = describe_afm_metric(channel_label, image)
    return display_image, metric_text, unit


def scale_afm_channel_for_display(channel_label: str, image: np.ndarray) -> tuple[np.ndarray, str]:
    """Scale one raw AFM/PFM channel into its default display unit."""
    unit = infer_afm_channel_unit(channel_label, image)
    return scale_afm_channel_for_unit(channel_label, image, unit), unit


def scale_afm_channel_for_unit(channel_label: str, image: np.ndarray, unit: str) -> np.ndarray:
    """Scale one raw AFM/PFM channel for display in ``unit``."""
    if unit == "deg":
        return np.asarray(image, dtype=float)

    values_m = _length_values_in_meters(channel_label, image)
    unit_scale = {"fm": 1e15, "pm": 1e12, "nm": 1e9, MICRON_UNIT: 1e6, "um": 1e6, "mm": 1e3}
    scale = unit_scale.get(unit)
    return values_m * scale if scale is not None else np.asarray(image, dtype=float)


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


def _length_values_in_meters(channel_label: str, image: np.ndarray) -> np.ndarray:
    """Return length-like channel values normalized to meters."""
    values = np.asarray(image, dtype=float)
    normalized = channel_label.strip().lower().replace(" ", "")
    if normalized == "latamplitude":
        return values * _lat_amplitude_meter_scale(values)
    return values


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


def should_show_metric_overlay(channel_label: str, *, multiple_plots: bool) -> bool:
    """Return whether the RMS label should be drawn for one channel."""
    if not multiple_plots:
        return True
    return channel_label.strip().lower() == "height"


def _add_metric_overlay(axis, metric_text: str) -> None:
    """Backward-compatible private alias for :func:`add_metric_overlay`."""
    add_metric_overlay(axis, metric_text)


def _compute_rms_metric(image: np.ndarray) -> float:
    """Backward-compatible private alias for :func:`compute_rms_metric`."""
    return compute_rms_metric(image)


def _describe_afm_metric(channel_label: str, image: np.ndarray) -> tuple[str, str]:
    """Backward-compatible private alias for :func:`describe_afm_metric`."""
    return describe_afm_metric(channel_label, image)


def _should_show_metric_overlay(channel_label: str, *, multiple_plots: bool) -> bool:
    """Backward-compatible private alias for :func:`should_show_metric_overlay`."""
    return should_show_metric_overlay(channel_label, multiple_plots=multiple_plots)


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


def show_pfm_images(imgs, labels=None, fig_name=None, cmap="viridis", cols: int = 3, **kwargs):
    """Show a simple grid of PFM/AFM image channels."""
    arr = np.asarray(imgs)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3:
        raise ValueError("imgs must be a 2D image or 3D image stack")

    n_images = arr.shape[2]
    labels = labels or [f"image {idx}" for idx in range(n_images)]
    fig, axes = layout_fig(n_images, mod=cols, figsize=kwargs.pop("figsize", None))
    for idx, ax in enumerate(axes):
        ax.imshow(arr[:, :, idx], cmap=cmap)
        ax.set_title(str(labels[idx]) if idx < len(labels) else f"image {idx}")
        ax.set_axis_off()
    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name, dpi=kwargs.pop("dpi", 300), bbox_inches="tight")
    return fig, axes


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
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (5, 4)))
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


class tip_potisition_analyzer:
    """Legacy spelling kept for compatibility with existing notebooks."""

    def show_tune(self, freq, amps, colors=None, positions=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
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


__all__ = [
    "AfmDataset",
    "AfmPreviewOptions",
    "AfmPreviewRender",
    "AFMVisualizer",
    "add_metric_overlay",
    "compute_rms_metric",
    "convert_with_unit",
    "default_preview_colorbar_setting",
    "default_preview_scalebar_setting",
    "describe_afm_metric",
    "df_scatter",
    "infer_afm_channel_unit",
    "load_afm_dataset",
    "plot_afm_channels",
    "prepare_afm_channel_display",
    "preferred_channel_index",
    "render_afm_preview",
    "scale_afm_channel_for_display",
    "scale_afm_channel_for_unit",
    "should_show_metric_overlay",
    "show_pfm_images",
    "tip_potisition_analyzer",
]
