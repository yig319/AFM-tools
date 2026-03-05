.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/yig319/AFM-tools.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/yig319/AFM-tools
    .. image:: https://readthedocs.org/projects/AFM-tools/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://AFM-tools.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/yig319/AFM-tools/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/yig319/AFM-tools
    .. image:: https://img.shields.io/pypi/v/AFM-tools.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/AFM-tools/
    .. image:: https://img.shields.io/conda/vn/conda-forge/AFM-tools.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/AFM-tools
    .. image:: https://pepy.tech/badge/AFM-tools/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/AFM-tools
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/AFM-tools

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=========
AFM-tools
=========


    A Python package for visualizing and analyzing Atomic Force Microscopy(AFM) and Piezoelectric Force Microscopy(PFM) experimental data, offering tools to process, visualize, and extract meaningful insights from AFM images and measurements.


This Python package provides a suite of tools for the visualization and analysis of Atomic Force Microscopy (AFM) experimental data. Designed for researchers working with AFM techniques, the package simplifies the processing of raw AFM data, including height, amplitude, and phase images. The built-in functionality allows users to visualize AFM scans in image and video if temporal dependent data, apply filters for noise reduction, and extract key metrics such as roughness, feature dimensions, and ferroelectric domain structures.

Key features:
- Support for multiple AFM data formats (e.g., .spm, .afm, .ibw).
- Real-time 2D and 3D visualization of AFM data.
- Data filtering and smoothing techniques.
- Tools for extracting quantitative measurements from AFM images.
- Customizable workflows for domain structure analysis.

This package is particularly useful for materials science researchers and AFM users who want to streamline data processing and explore advanced data processing and analysis.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.


mkinit --recursive --write /path/to/AFM-tools/src/afm_learn
