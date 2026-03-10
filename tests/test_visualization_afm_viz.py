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

    def _convert_with_unit(value):
        return f"{value:.2f}"

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


def test_show_pfm_images_saves_figure(afm_viz_module, no_show, tmp_path):
    imgs = np.random.rand(16, 16, 6)
    labels = [f"img_{i}" for i in range(6)]
    out_file = tmp_path / "pfm_grid.png"
    afm_viz_module.show_pfm_images(imgs, labels, fig_name=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


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
