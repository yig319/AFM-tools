from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest


def _load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


@pytest.fixture(scope="module")
def afm_viz_module():
    src_dir = Path(__file__).resolve().parents[1] / "src" / "afm_tools"

    pkg = types.ModuleType("afm_tools")
    pkg.__path__ = [str(src_dir)]
    sys.modules["afm_tools"] = pkg

    utils_stub = types.ModuleType("afm_tools.afm_utils")

    def _convert_scan_setting(scan_size):
        if isinstance(scan_size, dict):
            return scan_size
        return {"image_size": scan_size[0], "scale_size": scan_size[1], "units": scan_size[2]}

    def _convert_with_unit(value, unit="m"):
        if unit == "m":
            if abs(value) < 1e-9:
                return f"{value * 1e12:.2f} pm"
            if abs(value) < 1e-6:
                return f"{value * 1e9:.2f} nm"
            return f"{value * 1e6:.2f} µm"
        if unit == "deg":
            return f"{value:.2f} deg"
        if unit == "nm":
            return f"{value * 1e9:.2f} nm"
        if unit == "pm":
            return f"{value * 1e12:.2f} pm"
        return f"{value:.2f} {unit}".strip()

    def _define_percentage_threshold(image, percentage=(2, 98)):
        return np.percentile(image, percentage)

    def _format_func(value, unit=""):
        return f"{value:.2f}{unit}"

    utils_stub.convert_scan_setting = _convert_scan_setting
    utils_stub.convert_with_unit = _convert_with_unit
    utils_stub.define_percentage_threshold = _define_percentage_threshold
    utils_stub.format_func = _format_func
    sys.modules["afm_tools.afm_utils"] = utils_stub

    _load_module("afm_tools.viz_layout", src_dir / "viz_layout.py")

    domain_stub = types.ModuleType("afm_tools.domain_analysis")

    def _find_histogram_peaks(image, **kwargs):
        return np.array([float(np.median(image))]), np.array([image.size])

    domain_stub.find_histogram_peaks = _find_histogram_peaks
    sys.modules["afm_tools.domain_analysis"] = domain_stub

    module = _load_module("afm_tools.afm_viz", src_dir / "afm_viz.py")
    yield module

    for name in [
        "afm_tools.afm_viz",
        "afm_tools.domain_analysis",
        "afm_tools.viz_layout",
        "afm_tools.afm_utils",
        "afm_tools",
    ]:
        sys.modules.pop(name, None)


def test_afm_visualizer_viz_returns_axes_and_title(afm_viz_module):
    img = np.random.randn(32, 32)
    viz = afm_viz_module.AFMVisualizer(
        colorbar_setting={
            "colorbar_type": "percent",
            "colorbar_range": (2, 98),
            "outliers_std": None,
            "symmetric_clim": False,
            "visible": False,
        },
        zero_mean=False,
        scalebar=True,
    )
    fig, ax = viz.viz(
        img=img,
        scan_size={"image_size": 32, "scale_size": 8, "units": "nm"},
        title="height map",
    )
    assert fig is not None
    assert ax.get_title() == "height map"
    assert len(ax.texts) >= 1


def test_afm_visualizer_uses_compact_colorbar_by_default(afm_viz_module):
    img = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    viz = afm_viz_module.AFMVisualizer(
        colorbar_setting={
            "colorbar_type": "percent",
            "colorbar_range": (0, 100),
            "visible": True,
        },
        scalebar=False,
    )

    fig, _ax = viz.viz(img=img, cbar_unit="nm")
    colorbar_axis = fig.axes[-1]

    assert colorbar_axis.get_title() == "nm"
    assert colorbar_axis.yaxis.majorTicks[0]._tickdir == "in"
    assert colorbar_axis.yaxis.majorTicks[0].label1.get_fontsize() == 7


def test_afm_visualizer_can_use_matplotlib_colorbar_style(afm_viz_module):
    img = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    viz = afm_viz_module.AFMVisualizer(
        colorbar_setting={
            "colorbar_type": "percent",
            "colorbar_range": (0, 100),
            "visible": True,
            "style": "matplotlib",
        },
        scalebar=False,
    )

    fig, _ax = viz.viz(img=img, cbar_unit="nm")
    colorbar_axis = fig.axes[-1]

    assert colorbar_axis.get_title() == ""
    assert colorbar_axis.get_ylabel() == "nm"


def test_show_pfm_images_saves_figure(afm_viz_module, no_show, tmp_path):
    imgs = np.random.rand(16, 16, 6)
    labels = [f"img_{i}" for i in range(6)]
    out_file = tmp_path / "pfm_grid.png"
    afm_viz_module.show_pfm_images(imgs, labels, fig_name=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_afm_channels_accepts_loaded_dataset(afm_viz_module, no_show):
    dataset = types.SimpleNamespace(
        data=np.random.rand(16, 16, 3),
        labels=["Height", "Amplitude", "Phase"],
        scan_size_m={"image_size": 16, "scale_size": 5, "units": "µm"},
        sample="test_sample",
        path=Path("test_sample.ibw"),
    )

    fig, axes = afm_viz_module.plot_afm_channels(
        dataset,
        selected_channel_indices=[0, 2],
        n_cols=2,
        scalebar=False,
        colorbar_setting={"visible": False},
        show_sample_title=True,
    )

    assert fig is not None
    assert axes.shape == (1, 2)
    assert axes[0, 0].get_title() == "0: Height"
    assert axes[0, 1].get_title() == "2: Phase"
    assert fig._suptitle.get_text() == "test_sample"


def test_afm_metric_helpers_are_public(afm_viz_module):
    image = np.array([[0.0, 1.0], [2.0, 3.0]])

    assert afm_viz_module.compute_rms_metric(image) == pytest.approx(np.sqrt(1.25))
    metric_text, unit = afm_viz_module.describe_afm_metric("Phase", image)

    assert metric_text == "1.12 deg"
    assert unit == "deg"
    assert afm_viz_module.infer_afm_channel_unit("LatAmplitude", image * 1e-9) == "nm"
    assert afm_viz_module.infer_afm_channel_unit("Amplitude", image * 1e-12) == "pm"
    assert afm_viz_module.should_show_metric_overlay("Height", multiple_plots=True)
    assert not afm_viz_module.should_show_metric_overlay("Phase", multiple_plots=True)


def test_render_afm_preview_returns_status_and_overlay(afm_viz_module, no_show):
    height = np.linspace(0.0, 1e-9, 64).reshape(8, 8)
    phase = np.linspace(-2.0, 2.0, 64).reshape(8, 8)
    dataset = afm_viz_module.AfmDataset(
        file_path="demo.ibw",
        images=np.dstack([height, phase]),
        sample_name="demo",
        labels=["Height", "Phase"],
        scan_size={"image_size": 8, "scale_size": 2, "units": "µm"},
    )

    rendered = afm_viz_module.render_afm_preview(
        dataset,
        afm_viz_module.AfmPreviewOptions(
            selected_channel_indices=[0],
            show_metric_overlay=True,
        ),
    )

    assert rendered.figure is not None
    assert rendered.message.startswith("AFM preview updated for Height")
    assert any(text.get_text().startswith("RMS =") for text in rendered.figure.axes[0].texts)


def test_df_scatter_simple_mode_runs(afm_viz_module, no_show):
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [1.5, 2.5, 3.5],
            "label": ["a", "b", "c"],
        }
    )
    afm_viz_module.df_scatter(
        df1=df,
        df2=None,
        xaxis="x",
        yaxis="y",
        label_with="label",
        style="simple",
    )


def test_tip_position_show_tune_runs(afm_viz_module, no_show):
    analyzer = afm_viz_module.tip_potisition_analyzer()
    freq = np.linspace(30, 720, 100)
    amps = [np.sin(freq / 90.0), np.cos(freq / 120.0)]
    analyzer.show_tune(
        freq=freq,
        amps=amps,
        colors=["tab:blue", "tab:orange"],
        positions=["p1", "p2"],
    )
