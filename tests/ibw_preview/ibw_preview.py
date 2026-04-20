"""Render and test the bundled AFM/PFM IBW preview.

Run from the AFM-tools repository root:

    python tests/ibw_preview/ibw_preview.py --channels all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from matplotlib.transforms import Bbox
import numpy as np
import pytest

PREVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
SAMPLE_IBW = PREVIEW_DIR / "sample.ibw"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from afm_tools import afm_viz  # noqa: E402


def load_sample_dataset():
    """Load the bundled IBW sample used by both the example and tests."""
    if not SAMPLE_IBW.exists():
        pytest.skip("tests/ibw_preview/sample.ibw is not available")
    return afm_viz.load_afm_dataset(str(SAMPLE_IBW))


def render_sample_preview(
    *,
    channels: str = "preferred",
    colorbar_style: str = "compact",
    output: str | Path | None = None,
):
    """Render the sample IBW preview and save it to ``output`` when provided."""
    dataset = load_sample_dataset()
    channel_indices = parse_channels(channels, dataset.labels)
    rendered = afm_viz.render_afm_preview(
        dataset,
        afm_viz.AfmPreviewOptions(
            selected_channel_indices=channel_indices,
            show_metric_overlay=True,
            colorbar_setting={"style": colorbar_style},
        ),
    )

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        rendered.figure.savefig(output, dpi=300, bbox_inches="tight")
    return rendered


def parse_channels(value: str, labels: list[str]) -> list[int]:
    """Return channel indices from ``preferred``, ``all`` or ``0,1,2`` text."""
    if value.lower() == "preferred":
        return [afm_viz.preferred_channel_index(labels)]
    if value.lower() == "all":
        return list(range(len(labels)))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def default_output_path(channels: str) -> Path:
    """Return the default PNG output path for the selected channel mode."""
    suffix = "all" if channels.lower() == "all" else "preferred"
    return PREVIEW_DIR / "outputs" / f"sample_ibw_preview_{suffix}.png"


def _assert_compact_colorbar(figure, expected_unit):
    colorbar_axis = figure.axes[-1]
    assert colorbar_axis.get_title() == expected_unit
    assert colorbar_axis.get_ylabel() == ""
    assert colorbar_axis.yaxis.majorTicks[0]._tickdir == "in"
    assert colorbar_axis.yaxis.majorTicks[0].label1.get_fontsize() == 7


def _assert_scalebar_label_clear(axis):
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    patch_box = axis.patches[-1].get_window_extent(renderer)
    scale_text = next(text for text in axis.texts if "RMS =" not in text.get_text())
    text_box = scale_text.get_window_extent(renderer)
    assert not Bbox.overlaps(patch_box, text_box)


def test_sample_ibw_channel_units_follow_adaptive_length_rule():
    dataset = load_sample_dataset()
    units = {
        label: afm_viz.describe_afm_metric(label, dataset.images[:, :, index])[1]
        for index, label in enumerate(dataset.labels)
    }

    assert units["Height"] == "nm"
    assert units["Amplitude"] == "pm"
    if "LatAmplitude" in units:
        assert units["LatAmplitude"] == "nm"
    if "LatPhase" in units:
        assert units["LatPhase"] == "deg"
    assert units["Phase"] == "deg"
    assert units["ZSensor"] == "nm"


def test_amplitude_units_are_adaptive_not_label_fixed():
    pm_amplitude = np.array([[0.0, 5e-12], [1e-11, 5e-11]])
    nm_amplitude = np.array([[0.0, 5e-10], [1e-9, 5e-9]])
    um_amplitude = np.array([[0.0, 5e-7], [1e-6, 5e-6]])
    lat_amplitude_um = np.array([[0.0, 5e-4], [1e-3, 2e-3]])

    assert afm_viz.infer_afm_channel_unit("Amplitude", pm_amplitude) == "pm"
    assert afm_viz.infer_afm_channel_unit("Amplitude", nm_amplitude) == "nm"
    assert afm_viz.infer_afm_channel_unit("Amplitude", um_amplitude) == "\u00b5m"
    assert afm_viz.infer_afm_channel_unit("LatAmplitude", lat_amplitude_um) == "nm"


def test_sample_ibw_scalebar_uses_restored_physical_label():
    rendered = render_sample_preview(channels="preferred")
    axis = rendered.figure.axes[0]
    scale_text = next(text for text in axis.texts if "RMS =" not in text.get_text())

    assert scale_text.get_text() == "2 \u00b5m"
    _assert_scalebar_label_clear(axis)


def test_sample_ibw_render_preview_default_style(tmp_path):
    dataset = load_sample_dataset()
    channel_index = afm_viz.preferred_channel_index(dataset.labels)
    channel_label = dataset.labels[channel_index]
    _metric_text, colorbar_unit = afm_viz.describe_afm_metric(
        channel_label,
        dataset.images[:, :, channel_index],
    )

    rendered = render_sample_preview(
        channels="preferred",
        output=tmp_path / "sample_ibw_render_preview_default.png",
    )

    assert rendered.message.startswith(f"AFM preview updated for {channel_label}")
    assert any(text.get_text().startswith("RMS =") for text in rendered.figure.axes[0].texts)
    _assert_compact_colorbar(rendered.figure, colorbar_unit)
    _assert_scalebar_label_clear(rendered.figure.axes[0])


def test_sample_ibw_render_preview_can_use_matplotlib_colorbar_style():
    rendered = render_sample_preview(channels="preferred", colorbar_style="matplotlib")

    colorbar_axis = rendered.figure.axes[-1]
    assert colorbar_axis.get_title() == ""
    assert colorbar_axis.get_ylabel() != ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the bundled IBW preview using AFM-tools.")
    parser.add_argument(
        "--channels",
        default="preferred",
        help="'preferred', 'all', or comma-separated channel indices such as '0,1,2'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="PNG path to write. Defaults to tests/ibw_preview/outputs/.",
    )
    parser.add_argument(
        "--colorbar-style",
        choices=("compact", "matplotlib"),
        default="compact",
        help="Use the restored compact AFM style or Matplotlib's standard colorbar.",
    )
    parser.add_argument("--show", action="store_true", help="Open a Matplotlib window after saving.")
    args = parser.parse_args()

    output = Path(args.output) if args.output is not None else default_output_path(args.channels)
    rendered = render_sample_preview(
        channels=args.channels,
        colorbar_style=args.colorbar_style,
        output=output,
    )

    print(rendered.message)
    print(f"Saved preview to {output}")

    if args.show:
        from matplotlib import pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
