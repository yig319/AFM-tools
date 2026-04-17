=========
AFM-tools
=========

AFM-tools is a Python package for loading, processing, and visualizing Atomic
Force Microscopy (AFM) and Piezoelectric Force Microscopy (PFM) data.

Project Links
=============

- Source: https://github.com/yig319/AFM-tools
- Issues: https://github.com/yig319/AFM-tools/issues
- PyPI: https://pypi.org/project/AFM-tools/

Installation
============

Install from PyPI:

.. code-block:: bash

   pip install AFM-tools

Most users should use this pip install. It includes all core AFM/PFM features.
3D utilities in ``afm_tools.drawing_3d`` require ``mayavi`` (VTK/Qt stack),
which is recommended via Conda.

Install from source:

.. code-block:: bash

   git clone https://github.com/yig319/AFM-tools.git
   cd AFM-tools
   pip install -e .

Clone On A New Desktop (Core Pip Environment)
=============================================

From a fresh machine, this is the recommended setup for core AFM-tools usage:

.. code-block:: bash

   git clone https://github.com/yig319/AFM-tools.git
   cd AFM-tools
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements-dev.txt
   pip install -e .

Optional: 3D environment (Mayavi via Conda)
============================================

If you need ``drawing_3d``/Mayavi features:

.. code-block:: bash

   conda env create -f environment-mayavi.yml
   conda activate afm-tools-3d

This Conda environment installs ``mayavi``/``vtk``/``pyqt`` plus AFM-tools
dependencies. Use it when you need 3D visualization.

Quick Start
===========

.. code-block:: python

   import numpy as np
   from afm_tools.afm_viz import AFMVisualizer

   # Example image array (replace with real AFM/PFM image data)
   img = np.random.randn(256, 256)

   viz = AFMVisualizer()
   viz.viz(img=img, scan_size={"image_size": 256, "scale_size": 1, "units": "µm"})

IBW Preview Style
=================

The high-level IBW preview keeps the compact AFM/PFM plotting style as the
default: percentile color limits, MAD-based outlier handling, inward colorbar
ticks, small colorbar labels, the unit above the colorbar, and a bottom-right
scale bar with its label offset from the bar.

.. code-block:: python

   from afm_tools.afm_viz import AfmPreviewOptions, load_afm_dataset, render_afm_preview

   dataset = load_afm_dataset("tests/ibw_preview/sample.ibw")
   rendered = render_afm_preview(
       dataset,
       AfmPreviewOptions(selected_channel_indices=[0], show_metric_overlay=True),
   )
   rendered.figure.savefig("sample_preview.png", dpi=180, bbox_inches="tight")

You can tune the restored defaults without rewriting the plotting function:

.. code-block:: python

   options = AfmPreviewOptions(
       selected_channel_indices=[0],
       colorbar_setting={
           "style": "compact",
           "tick_labelsize": 7,
           "unit_position": "top",
           "scale_image": True,
       },
       scalebar_setting={
           "text_offset": 0.55,
           "text_fontsize": 9,
       },
   )

Use ``colorbar_setting={"style": "matplotlib"}`` for Matplotlib's standard
side-label colorbar. Use ``{"scale_image": False, "tick_unit": True}`` to keep
raw meter-valued image data and scale only the colorbar tick labels, matching
older AFM visualizer behavior.

To generate preview PNGs from the bundled sample IBW file:

.. code-block:: bash

   python tests/ibw_preview/ibw_preview.py
   python tests/ibw_preview/ibw_preview.py --channels all

The images are written to ``tests/ibw_preview/outputs`` by default.

Features
========

- Read and parse AFM-related wave/image formats.
- 2D/3D visualization utilities for AFM/PFM datasets.
- Domain and morphology analysis helpers.
- Video and plotting utilities for time/scan series.

Documentation
=============

Sphinx documentation is provided in the ``docs`` directory.

Build docs locally:

.. code-block:: bash

   pip install -r docs/requirements.txt
   pip install -e .
   sphinx-build -b html docs docs/_build/html

License
=======

This project is licensed under the MIT License. See ``LICENSE.txt``.
