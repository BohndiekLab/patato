#  Copyright (c) Thomas Else 2023.
#  License: MIT

"""
Unmixing module
======================

This module provides functions for commonly-used spectral analysis procedures in photoacoustic imaging.
Additionally, the module provides commonly-used spectra for chromophores in biological tissue.
"""


def get_default_unmixing_preset():
    """

    Returns
    -------

    """
    import os
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../unmixing/unmix_presets")
    return os.path.join(root_folder, "haemoglobin.json")
