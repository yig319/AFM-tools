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
   viz.viz(img=img, scan_size={"image_size": 256, "scale_size": 1, "units": "um"})

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
