============
Contributing
============

Thanks for contributing to AFM-tools.

Report Bugs or Request Features
===============================

- Search existing issues first: https://github.com/yig319/AFM-tools/issues
- Open a new issue with:
  - operating system and Python version
  - clear reproduction steps
  - expected vs actual behavior

Development Setup
=================

.. code-block:: bash

   git clone https://github.com/yig319/AFM-tools.git
   cd AFM-tools
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements-dev.txt
   pip install -e .

Run checks:

.. code-block:: bash

   pytest
   tox -e docs

Code Style
==========

- Keep changes focused and minimal.
- Add/update tests for behavior changes.
- Add docstrings for public APIs.
- Update ``README.rst`` and ``CHANGELOG.rst`` when needed.

Pull Request Process
====================

1. Create a feature branch.
2. Commit with clear messages.
3. Push and open a pull request.
4. Ensure CI is green before merge.

Release Process
===============

Publishing is automated by GitHub Actions on push to ``main``.
The workflow builds the package and publishes to PyPI using Trusted Publishing.

Before release:

- verify versioning strategy
- run tests locally
- confirm package metadata in ``setup.cfg``
