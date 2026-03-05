"""
    Setup file for AFM-tools.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.6.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

from pathlib import Path

from setuptools import setup


def _read_requirements(path: str) -> list[str]:
    req_path = Path(__file__).parent / path
    requirements = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r"):
            raise ValueError("Nested requirements files are not supported for install_requires")
        requirements.append(line)
    return requirements


if __name__ == "__main__":
    try:
        setup(
            install_requires=_read_requirements("requirements.txt"),
            use_scm_version={"version_scheme": "no-guess-dev"},
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
