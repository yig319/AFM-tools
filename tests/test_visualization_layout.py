from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


def _load_viz_layout():
    module_path = Path(__file__).resolve().parents[1] / "src" / "afm_learn" / "viz_layout.py"
    spec = spec_from_file_location("viz_layout_local", module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layout_fig_subplots_returns_expected_axes_count():
    mod = _load_viz_layout()
    fig, axes = mod.layout_fig(graph=3, mod=2, subplot_style="subplots", figsize=(6, 4))
    assert fig is not None
    assert len(axes) == 3


def test_layout_fig_gridspec_returns_expected_axes_count():
    mod = _load_viz_layout()
    fig, axes = mod.layout_fig(graph=4, mod=2, subplot_style="gridspec", figsize=(6, 6))
    assert fig is not None
    assert len(axes) == 4


def test_scalebar_adds_patch_and_text():
    mod = _load_viz_layout()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.random.rand(32, 32))
    mod.scalebar(ax, image_size=32, scale_size=8, units="nm", loc="br")
    assert len(ax.patches) >= 1
    assert len(ax.texts) >= 1


def test_scalebar_invalid_location_raises():
    mod = _load_viz_layout()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.random.rand(32, 32))
    with pytest.raises(ValueError):
        mod.scalebar(ax, image_size=32, scale_size=8, units="nm", loc="invalid")
