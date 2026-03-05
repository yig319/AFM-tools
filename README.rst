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

Install from source:

.. code-block:: bash

   git clone https://github.com/yig319/AFM-tools.git
   cd AFM-tools
   pip install -e .

Quick Start
===========

.. code-block:: python

   import numpy as np
   from afm_learn.afm_viz import AFMVisualizer

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
