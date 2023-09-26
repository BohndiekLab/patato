[![Journal of open source software status](https://joss.theoj.org/papers/456eaf591244858915ad8730dcbc19d7/status.svg)](https://joss.theoj.org/papers/456eaf591244858915ad8730dcbc19d7)
[![Documentation Status](https://readthedocs.org/projects/patato/badge/?version=latest)](https://patato.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/bohndieklab/patato/blob/main/LICENSE.MD)
[![PyPI version](https://badge.fury.io/py/patato.svg)](https://badge.fury.io/py/patato)
[![Build](https://github.com/bohndieklab/patato/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/bohndieklab/patato/actions/workflows/build_wheels.yml)

![Logo](https://github.com/BohndiekLab/patato/raw/main/docs/logos/PATATO%20Logo_1_Combination.png "Logo")

# PATATO: PhotoAcoustic Tomography Analysis TOolkit

[Documentation](https://patato.readthedocs.io/en/develop/)

PATATO is an Open-Source project to enable the analysis of photoacoustic (PA) imaging data in a transparent, reproducible and extendable way. We provide efficient, GPU-optimised implementations of common PA algorithms written around standard Python libraries, including filtered backprojection, model-based reconstruction and spectral unmixing.

The tool supports many file formats, such as the International Photoacoustic Standardisation Consortium (IPASC) data format, and it can be extended to support custom data formats. We hope that this toolkit can enable faster and wider dissemination of analysis techniques for PA imaging and provide a useful tool to the community.

* Please report any bugs or issues you find to our GitHub repository
* Please do get involved! Contact Thomas Else (thomas.else@cruk.cam.ac.uk).

## Getting Started
In order to use PATATO, you must have a Python environment set up on your computer. We recommend using Anaconda (http://anaconda.com) to run Python, particularly if you are using Windows. You may wish to setup a separate Anaconda environment to install PATATO to minimise conflicts between dependency versions.

```shell
pip install --upgrade pip setuptools
pip install patato
```

To setup support for image reconstruction on Windows, or for GPU support, please follow the installation guide in the documentation.

## Documentation, examples and contributing
Documentation for PATATO can be found at https://patato.readthedocs.io/en/latest/?badge=latest.

Copyright (c) Thomas Else 2022-23.
Distributed under a MIT License.
