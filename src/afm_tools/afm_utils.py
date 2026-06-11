"""Core AFM/PFM file loading and small unit helpers.

This module is the foundation layer for AFM-tools. It owns IBW parsing,
channel access, scan-size parsing, and unit formatting. Downstream notebooks
should import these functions instead of reimplementing file-reader details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AFMImage:
    """Parsed AFM/PFM image stack from an Igor Binary Wave file.

    Attributes
    ----------
    data:
        2D image or 3D stack with channels on the last axis.
    labels:
        Channel labels after AFM-tools applies common Asylum channel ordering.
    scan_size_m:
        Physical scan size in meters when it can be parsed from the IBW note.
    notes:
        Key-value metadata parsed from the IBW note string.
    """

    data: np.ndarray
    path: Path
    sample: str
    labels: list[str]
    scan_size_m: float | None
    note: str
    notes: dict[str, str]


# Decode raw bytes to latin-1 string (handles mixed bytes/str from Igor binary reader).
def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="ignore")
    return str(value)


# Try to import an Igor Binary Wave reader from bundled or external packages.
def _binarywave_loader():
    try:
        from afm_tools.igor import igor as binarywave

        return binarywave
    except Exception:
        try:
            from afm_tools.igor import binarywave

            return binarywave
        except Exception:
            try:
                from igor import binarywave

                return binarywave
            except Exception as exc:
                try:
                    from igor2 import binarywave

                    return binarywave
                except Exception as exc2:
                    raise ImportError(
                        "Install AFM-tools with its Igor reader, or install igor/igor2, "
                        "to read .ibw files."
                    ) from exc2


# Parse the free-form Igor note text into a dict of key-value metadata strings.
def parse_notes(note: str) -> dict[str, str]:
    """Parse the free-form Igor note text into a dictionary of metadata."""
    pattern = r"([\w\s]+):\s*([^\r]+)"
    notes: dict[str, str] = {}
    counter: dict[str, int] = {}
    for key, value in re.findall(pattern, note):
        key = re.sub(r"\r", "", key).strip()
        if key in counter:
            counter[key] += 1
            key = f"{key}_{counter[key]}"
        else:
            counter[key] = 0
        notes[key] = value.strip()
    return notes


# Extract the physical scan size in meters from parsed notes, falling back to regex on raw note text.
def _parse_scan_size_m(notes: dict[str, str], note: str) -> float | None:
    raw = notes.get("ScanSize") or notes.get("Scan Size") or notes.get("FastScanSize")
    if raw is not None:
        try:
            return _value_with_unit_to_m(raw)
        except ValueError:
            pass

    for pattern in (
        r"ScanSize\s*:\s*([-+0-9.eE]+)\s*([A-Za-z]*)",
        r"Scan Size\s*:\s*([-+0-9.eE]+)\s*([A-Za-z]*)",
        r"FastScanSize\s*:\s*([-+0-9.eE]+)\s*([A-Za-z]*)",
    ):
        match = re.search(pattern, note)
        if match:
            return _value_with_unit_to_m(" ".join(group for group in match.groups() if group))
    return None


# Convert a value-with-unit string (e.g. "5 um", "10 nm") to metres.
def _value_with_unit_to_m(text: str) -> float:
    match = re.search(r"([-+0-9.eE]+)\s*([A-Za-z]*)", str(text))
    if not match:
        raise ValueError(f"Could not parse numeric value from {text!r}")
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"nm", "nanometer", "nanometers"}:
        return value * 1e-9
    if unit in {"um", "micron", "microns"}:
        return value * 1e-6
    if unit == "mm":
        return value * 1e-3
    return value


# Infer channel labels from wave metadata, applying known Asylum PFM/AC-mode label reordering.
def _labels_from_wave(wave, notes: dict[str, str], data: np.ndarray, mode: str | None) -> list[str]:
    labels_current: list[str] = []
    try:
        for raw in wave["labels"][-2][1:]:
            label = _decode(raw)
            for marker in ("Retrace", "Trace"):
                index = label.find(marker)
                if index >= 0:
                    label = label[:index]
            label = label.strip()
            if label:
                labels_current.append(label)
    except Exception:
        labels_current = []

    labels_correct: list[str] | None = None
    mode = mode or notes.get("ImagingMode")
    if mode == "PFM Mode":
        if len(labels_current) == 6:
            labels_correct = [
                notes.get("Channel1DataType_1", labels_current[0]),
                notes.get("Channel2DataType", labels_current[1]),
                notes.get("Channel3DataType", labels_current[2]),
                notes.get("Channel4DataType", labels_current[3]),
                notes.get("Channel5DataType", labels_current[4]),
                notes.get("Channel6DataType", labels_current[5]),
            ]
            if labels_correct == ["LatAmplitude", "LatPhase", "Height", "Amplitude", "Phase", "ZSensor"]:
                labels_correct = ["Height", "LatAmplitude", "LatPhase", "ZSensor", "Amplitude", "Phase"]
        elif len(labels_current) == 4:
            labels_correct = ["Height", "Deflection", "Amplitude", "Phase"]
    elif mode == "AC Mode" and len(labels_current) >= 4:
        labels_correct = ["Height", "Amplitude", "Phase", "ZSensor"]

    if labels_correct is not None and labels_current:
        labels_available = [label for label in labels_correct if label in labels_current]
        if len(labels_available) == len(labels_correct):
            return labels_correct

    if data.ndim == 3:
        if not labels_current:
            labels_current = [f"channel_{i}" for i in range(data.shape[2])]
        elif len(labels_current) < data.shape[2]:
            labels_current.extend([f"channel_{i}" for i in range(len(labels_current), data.shape[2])])
        return labels_current[: data.shape[2]]
    return labels_current or ["height"]


# Load an Asylum/Igor .ibw AFM/PFM image stack into a named AFMImage container.
def load_ibw(path: str | Path, mode: str | None = None, reorder_channels: bool = True) -> AFMImage:
    """Load an Asylum/Igor ``.ibw`` AFM/PFM image stack.

    Parameters
    ----------
    path:
        IBW file path.
    mode:
        Optional imaging-mode hint. When omitted, AFM-tools reads the mode from
        the file note when possible.
    reorder_channels:
        Reorder known PFM/AC-mode channel layouts into a consistent label order.

    Returns
    -------
    AFMImage
        A named container with image data, labels, scan size, and metadata.
    """
    path = Path(path)
    obj = _binarywave_loader().load(str(path))
    wave = obj["wave"]
    data = np.asarray(wave["wData"])
    if data.ndim >= 2:
        data = np.swapaxes(data, 0, 1)
        data = np.flip(data, 0)

    try:
        sample = _decode(wave["wave_header"]["bname"]).strip() or path.stem
    except Exception:
        sample = path.stem

    note = _decode(wave.get("note", b""))
    notes = parse_notes(note)
    labels = _labels_from_wave(wave, notes, data, mode)

    if reorder_channels and data.ndim == 3:
        try:
            raw_labels = []
            for raw in wave["labels"][-2][1:]:
                label = _decode(raw)
                for marker in ("Retrace", "Trace"):
                    index = label.find(marker)
                    if index >= 0:
                        label = label[:index]
                raw_labels.append(label.strip())
            if raw_labels and set(labels).issubset(raw_labels):
                indices = [raw_labels.index(label) for label in labels]
                data = data[:, :, indices]
        except Exception:
            pass

    return AFMImage(
        data=data.astype(np.float32, copy=False),
        path=path,
        sample=sample,
        labels=labels,
        scan_size_m=_parse_scan_size_m(notes, note),
        note=note,
        notes=notes,
    )


# Legacy tuple-API wrapper around load_ibw: returns (images, sample_name, labels, scan_size_m).
def parse_ibw(file: str | Path, mode: str | None = None):
    """Return the legacy tuple API: ``(images, sample_name, labels, scan_size_m)``.

    This is kept because PLD_workflow and older notebooks use this compact
    tuple form. New code can use :func:`load_ibw` for named fields.
    """
    image = load_ibw(file, mode=mode)
    return image.data, image.sample, image.labels, image.scan_size_m


# Return one 2D channel from an AFMImage or raw numpy stack, by label substring or index.
def get_channel(image: AFMImage | np.ndarray, index: int = 0, label_contains: str | None = None) -> np.ndarray:
    """Return one channel from an :class:`AFMImage` or raw image stack."""
    data = image.data if isinstance(image, AFMImage) else np.asarray(image)
    labels = image.labels if isinstance(image, AFMImage) else []
    if data.ndim == 2:
        return data
    if label_contains:
        needle = label_contains.lower()
        for i, label in enumerate(labels):
            if needle in label.lower():
                return data[:, :, i]
    return data[:, :, index]


# Return (low, high) percentile limits from the finite pixels of an image — used for colorbar clipping.
def define_percentage_threshold(image: np.ndarray, percentage=(2, 98)) -> tuple[float, float]:
    """Return finite-data percentile limits for image display."""
    return tuple(np.percentile(np.asarray(image)[np.isfinite(image)], percentage))


MICRON_UNIT = "\u00b5m"


# Normalize scan-size inputs (meters, tuple, or dict) into a dict for scale-bar drawing.
def convert_scan_setting(scan_size):
    """Normalize scan-size inputs for scale-bar drawing.

    Numeric scan sizes are in meters. The returned ``image_size`` and
    ``scale_size`` are physical lengths in the returned unit, matching the
    original hand-tuned AFM visualizer contract.
    """
    if isinstance(scan_size, dict):
        return scan_size
    if isinstance(scan_size, (tuple, list)) and len(scan_size) == 3:
        return {"image_size": scan_size[0], "scale_size": scan_size[1], "units": scan_size[2]}
    scan_size = flexible_round(float(scan_size))

    scale_ranges = [
        (2e-5, 5e-5, {"scale_size": 5, "units": MICRON_UNIT}),
        (1e-5, 2e-5, {"scale_size": 2, "units": MICRON_UNIT}),
        (3e-6, 1e-5, {"scale_size": 1, "units": MICRON_UNIT}),
        (2e-6, 3e-6, {"scale_size": 500, "units": "nm"}),
        (1e-6, 2e-6, {"scale_size": 200, "units": "nm"}),
        (5e-7, 1e-6, {"scale_size": 100, "units": "nm"}),
        (2e-7, 5e-7, {"scale_size": 50, "units": "nm"}),
        (1e-7, 2e-7, {"scale_size": 20, "units": "nm"}),
        (5e-8, 1e-7, {"scale_size": 10, "units": "nm"}),
        (3e-8, 5e-8, {"scale_size": 5, "units": "nm"}),
        (2e-8, 3e-8, {"scale_size": 3, "units": "nm"}),
        (1e-8, 2e-8, {"scale_size": 2, "units": "nm"}),
        (5e-9, 1e-8, {"scale_size": 1, "units": "nm"}),
        (3e-9, 5e-9, {"scale_size": 500, "units": "pm"}),
        (2e-9, 3e-9, {"scale_size": 300, "units": "pm"}),
        (1e-9, 2e-9, {"scale_size": 200, "units": "pm"}),
        (5e-10, 1e-9, {"scale_size": 100, "units": "pm"}),
        (3e-10, 5e-10, {"scale_size": 50, "units": "pm"}),
        (2e-10, 3e-10, {"scale_size": 30, "units": "pm"}),
        (1e-10, 2e-10, {"scale_size": 20, "units": "pm"}),
        (1e-11, 1e-10, {"scale_size": 10, "units": "pm"}),
    ]
    unit_scale = {"pm": 1e-12, "nm": 1e-9, MICRON_UNIT: 1e-6, "um": 1e-6, "mm": 1e-3}

    for min_val, max_val, scale_params in scale_ranges:
        if min_val <= abs(scan_size) <= max_val:
            setting = dict(scale_params)
            setting["image_size"] = scan_size / unit_scale[setting["units"]]
            return setting
    return {"image_size": scan_size, "scale_size": scan_size, "units": "m"}


# Round a value to a given number of significant digits while preserving order of magnitude.
def flexible_round(value, sig_digits=1):
    """Round a value by significant digits while preserving order of magnitude."""
    value = float(value)
    if value == 0:
        return value
    magnitude = np.floor(np.log10(abs(value)))
    factor = 10**magnitude
    return round(value / factor, sig_digits) * factor


_LENGTH_UNIT_SCALE = {
    "fm": 1e-15,
    "pm": 1e-12,
    "nm": 1e-9,
    MICRON_UNIT: 1e-6,
    "um": 1e-6,
    "mm": 1e-3,
    "m": 1.0,
}


# Format a value (stored in metres) to a human-readable engineering-unit string (pm, nm, µm, mm).
def convert_with_unit(value: float, unit: str = "m") -> str:
    """Format a value with the compact AFM engineering unit style.

    Length values stored in meters are shown with ``pm``, ``nm``, ``um`` or
    ``mm`` when possible. Passing an explicit unit scales to that unit, which
    keeps metric text consistent with colorbar units.
    """

    value = float(value)
    unit = unit or "m"
    if unit in {"m", "meter", "meters"}:
        for suffix, scale in (("pm", 1e-12), ("nm", 1e-9), (MICRON_UNIT, 1e-6), ("mm", 1e-3)):
            scaled = value / scale
            if abs(scaled) < 1000:
                return f"{scaled:.2f} {suffix}"
        return f"{value:.2e} m"

    scale = _LENGTH_UNIT_SCALE.get(unit)
    if scale is not None:
        return f"{value / scale:.2f} {unit}"
    return f"{value:.2f} {unit}".strip()


# Format compact colorbar tick text — scales values from metres to the given display unit.
def format_func(value: float, unit: str = "") -> str:
    """Format compact colorbar tick text.

    When ``unit`` is a length unit, values are scaled from meters into that
    unit. When ``unit`` is empty, the value is formatted as-is; this is useful
    after an image has already been scaled for display.
    """

    scale = _LENGTH_UNIT_SCALE.get(unit)
    scaled = float(value) / scale if scale is not None else float(value)
    if unit == "deg":
        scaled = float(value)
    if 0 < abs(scaled) < 0.01 or abs(scaled) >= 10000:
        return f"{scaled:.1e}"
    if abs(scaled) >= 100:
        return f"{scaled:.1f}".rstrip("0").rstrip(".")
    if abs(scaled) >= 10:
        return f"{scaled:.1f}"
    if abs(scaled) >= 1 or scaled == 0:
        return f"{scaled:.2f}".rstrip("0").rstrip(".")
    return f"{scaled:.2g}"
