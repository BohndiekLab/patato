# PATATO: PhotoAcoustic Tomography Analysis TOolkit
[![Documentation Status](https://readthedocs.org/projects/patato/badge/?version=latest)](https://patato.readthedocs.io/en/latest/?badge=latest)

PATATO is an Open-Source project to enable the analysis of photoacoustic (PA) imaging data in a transparent, reproducible and extendable way. We provide efficient, GPU-optimised implementations of common PA algorithms written around standard Python libraries, including filtered backprojection, model-based reconstruction and spectral unmixing.

The tool supports many file formats, such as the International Photoacoustic Standardisation Consortium (IPASC) data format, and it can be extended to support custom data formats. We hope that this toolkit can enable faster and wider dissemination of analysis techniques for PA imaging and provide a useful tool to the community.

* Please report any bugs or issues you find to our GitHub repository
* Please do get involved! Contact Thomas Else (thomas.else@cruk.cam.ac.uk).

## Getting Started
In order to use PATATO, you must have a Python environment set up on your computer. We recommend using Anaconda (http://anaconda.com) to run Python, particularly if you are using Windows. You may wish to setup a separate Anaconda environment to install PATATO to minimise conflicts between dependency versions.

```shell
pip install git+https://github.com/tomelse/MSOTAnalysis
```

## Documentation, examples and contributing
Documentation for PATATO can be found at https://patato.readthedocs.io/en/latest/?badge=latest.

Copyright (c) Thomas Else 2022-23.
Distributed under a BSD-3 License.
