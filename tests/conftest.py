"""Shared pytest fixtures for AFM-tools."""

import matplotlib
import pytest

# Use a non-interactive backend so tests run in headless CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures():
    """Ensure figures do not leak between tests."""
    yield
    plt.close("all")


@pytest.fixture
def no_show(monkeypatch):
    """Disable ``plt.show()`` popups during tests."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
