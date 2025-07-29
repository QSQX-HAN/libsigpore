"""
    Setup file for proj_het.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.0.2.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
from setuptools import setup

with open("requirements.txt") as f:
    required = [x for x in f.read().splitlines() if not x.startswith("#")]

if __name__ == "__main__":
    try:
        setup(use_scm_version={"version_scheme": "no-guess-dev"}, install_requires=required)
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
