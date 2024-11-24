# PATATO: Photoacoustic Tomography Analysis Toolkit

[![Journal of open source software status](https://joss.theoj.org/papers/456eaf591244858915ad8730dcbc19d7/status.svg)](https://joss.theoj.org/papers/456eaf591244858915ad8730dcbc19d7)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/bohndieklab/patato/blob/main/LICENSE.MD)

<!-- [![Actions Status][actions-badge]][actions-link] -->

[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

![Logo](https://github.com/BohndiekLab/patato/raw/main/docs/logos/PATATO%20Logo_1_Combination.png "Logo")

PATATO is an Open-Source project to enable the analysis of photoacoustic (PA)
imaging data in a transparent, reproducible and extendable way. We provide
efficient, GPU-optimised implementations of common PA algorithms written around
standard Python libraries, including filtered backprojection, model-based
reconstruction and spectral unmixing.

The tool supports many file formats, such as the International Photoacoustic
Standardisation Consortium (IPASC) data format, and it can be extended to
support custom data formats. We hope that this toolkit can enable faster and
wider dissemination of analysis techniques for PA imaging and provide a useful
tool to the community.

- Please report any bugs or issues you find to our GitHub repository
- Please do get involved! Contact Thomas Else (thomas.else@cruk.cam.ac.uk).

## Getting Started

In order to use PATATO, you must have a Python environment set up on your
computer. We recommend using Anaconda (http://anaconda.com) to run Python,
particularly if you are using Windows. You may wish to setup a separate Anaconda
environment to install PATATO to minimise conflicts between dependency versions.

To setup support for image reconstruction on Windows, or for GPU support, please
follow the installation guide in the documentation.

## Citing PATATO

To cite PATATO, please reference our article in the Journal of Open Source
software,
[here](https://joss.theoj.org/papers/456eaf591244858915ad8730dcbc19d7).

## Documentation, examples and contributing

Documentation for PATATO can be found at
https://patato.readthedocs.io/en/latest/?badge=latest.

Copyright (c) Thomas Else 2022-24. Distributed under a MIT License.

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/BohndiekLab/PATATO/workflows/CI/badge.svg
[actions-link]:             https://github.com/BohndiekLab/PATATO/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/PATATO
[conda-link]:               https://github.com/conda-forge/PATATO-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/BohndiekLab/PATATO/discussions
[pypi-link]:                https://pypi.org/project/PATATO/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/PATATO
[pypi-version]:             https://img.shields.io/pypi/v/PATATO
[rtd-badge]:                https://readthedocs.org/projects/PATATO/badge/?version=latest
[rtd-link]:                 https://PATATO.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
