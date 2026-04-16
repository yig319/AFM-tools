"""Regression tests for real IBW preview rendering.

These tests exercise AFM-tools without PLD_workflow in the middle, so preview
style changes can be checked in this package before updating downstream apps.
"""

from pathlib import Path
import sys

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from afm_tools import afm_viz  # noqa: E402


SAMPLE_IBW = Path(__file__).with_name("sample.ibw")


def _load_sample_dataset():
    if not SAMPLE_IBW.exists():
        pytest.skip("tests/sample.ibw is not available")
    return afm_viz.load_afm_dataset(str(SAMPLE_IBW))


def _assert_compact_colorbar(figure, expected_unit):
    colorbar_axis = figure.axes[-1]
    assert colorbar_axis.get_title() == expected_unit
    assert colorbar_axis.get_ylabel() == ""
    assert colorbar_axis.yaxis.majorTicks[0]._tickdir == "in"
    assert colorbar_axis.yaxis.majorTicks[0].label1.get_fontsize() == 7


def test_sample_ibw_afm_visualizer_default_style(tmp_path):
    dataset = _load_sample_dataset()
    channel_index = afm_viz.preferred_channel_index(dataset.labels)
    channel_label = dataset.labels[channel_index]
    image = dataset.images[:, :, channel_index]
    _metric_text, colorbar_unit = afm_viz.describe_afm_metric(channel_label, image)

    visualizer = afm_viz.AFMVisualizer(
        colorbar_setting=afm_viz.default_preview_colorbar_setting(),
        scalebar=True,
    )
    figure, axis = visualizer.viz(
        image,
        scan_size=dataset.scan_size,
        title=channel_label,
        cbar_unit=colorbar_unit,
    )

    assert axis.get_title() == channel_label
    _assert_compact_colorbar(figure, colorbar_unit)

    output_path = tmp_path / "sample_ibw_afm_visualizer_default.png"
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    assert output_path.stat().st_size > 0


def test_sample_ibw_render_preview_default_style(tmp_path):
    dataset = _load_sample_dataset()
    channel_index = afm_viz.preferred_channel_index(dataset.labels)
    channel_label = dataset.labels[channel_index]
    _metric_text, colorbar_unit = afm_viz.describe_afm_metric(
        channel_label,
        dataset.images[:, :, channel_index],
    )

    rendered = afm_viz.render_afm_preview(
        dataset,
        afm_viz.AfmPreviewOptions(
            selected_channel_indices=[channel_index],
            show_metric_overlay=True,
        ),
    )

    assert rendered.message.startswith(f"AFM preview updated for {channel_label}")
    assert any(text.get_text().startswith("RMS =") for text in rendered.figure.axes[0].texts)
    _assert_compact_colorbar(rendered.figure, colorbar_unit)

    output_path = tmp_path / "sample_ibw_render_preview_default.png"
    rendered.figure.savefig(output_path, dpi=120, bbox_inches="tight")
    assert output_path.stat().st_size > 0


def test_sample_ibw_render_preview_can_use_matplotlib_colorbar_style():
    dataset = _load_sample_dataset()
    channel_index = afm_viz.preferred_channel_index(dataset.labels)
    channel_label = dataset.labels[channel_index]
    _metric_text, colorbar_unit = afm_viz.describe_afm_metric(
        channel_label,
        dataset.images[:, :, channel_index],
    )

    rendered = afm_viz.render_afm_preview(
        dataset,
        afm_viz.AfmPreviewOptions(
            selected_channel_indices=[channel_index],
            colorbar_setting={"style": "matplotlib"},
        ),
    )

    colorbar_axis = rendered.figure.axes[-1]
    assert rendered.message.startswith(f"AFM preview updated for {channel_label}")
    assert colorbar_axis.get_title() == ""
    assert colorbar_axis.get_ylabel() == colorbar_unit

