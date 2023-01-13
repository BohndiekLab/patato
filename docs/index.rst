PATATO: PhotoAcoustic Tomography Analysis TOolkit
=====================================================================

.. image:: https://readthedocs.org/projects/patato/badge/?version=latest
    :target: https://patato.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/tomelse/patato/blob/main/LICENSE.MD
    :alt: BSD 3-Clause License
.. image:: https://badge.fury.io/py/patato.svg
    :target: https://badge.fury.io/py/patato
    :alt: PyPI version

PATATO is an Open-Source project to enable the analysis of photoacoustic (PA)
imaging data in a transparent, reproducible and extendable way.
We provide efficient, GPU-optimised implementations of common PA
algorithms written around standard Python libraries, including
filtered backprojection, model-based reconstruction and spectral
unmixing.

The tool supports many file formats, such as the International
Photoacoustic Standardisation Consortium (IPASC) data format,
and it can be extended to support custom data formats. We hope
that this toolkit can enable faster and wider dissemination of
analysis techniques for PA imaging and provide a useful tool to
the community.

* Please report any bugs or issues you find to our GitHub repository
* Please do get involved! Contact Thomas Else (thomas.else@cruk.cam.ac.uk).

.. only:: latex

    User Guide
    """""""""""""""""""""

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :glob:

   intro/*

.. only:: latex

    Examples
    """""""""""""""""""""

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :glob:

   examples/*

.. only:: latex

    API Reference
    """""""""""""""""""""

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   Analysis Code <api/analysis_code>
   Convenience Scripts <api/convenience_scripts>
   Bohndiek-Lab Specific Code <api/bohndieklab_utilities>

.. only:: latex

    Developers
    """"""""""""""""""""

.. toctree::
   :maxdepth: 2
   :caption: Developers
   :glob:

   dev/*
