"""Utilities for AFM/PFM image parsing, analysis, and visualization."""

try:
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("AFM-tools")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from afm_tools.afm_utils import *  # noqa: F401,F403,E402
from afm_tools.afm_image_analyzer import *  # noqa: F401,F403,E402
from afm_tools.afm_viz import *  # noqa: F401,F403,E402
from afm_tools.domain_analysis import *  # noqa: F401,F403,E402
