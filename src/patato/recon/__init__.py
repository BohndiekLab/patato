#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from typing import Sequence

from .backprojection_opencl import OpenCLBackprojection
from .backprojection_reference import ReferenceBackprojection
from .reconstruction_algorithm import ReconstructionAlgorithm
from ..io.msot_data import PAData

RECONSTRUCTION_METHODS: Sequence[type(ReconstructionAlgorithm)] = [OpenCLBackprojection,
                                                                   ReferenceBackprojection]

RECONSTRUCTION_NAMES = {x.get_algorithm_name(): x for x in RECONSTRUCTION_METHODS}


def get_default_recon_preset(data: PAData):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """
    import numpy as np
    import os
    try:
        import pyopencl as cl
        CPU = not any(len(platform.get_devices(device_type=cl.device_type.GPU)) > 0 for platform in cl.get_platforms())
    except ImportError:
        cl = None
        CPU = False  # Should this maybe be true?
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "recon_presets")
    preset = ""

    if np.isclose(np.sqrt(np.sum(data.get_scan_geometry()[0] ** 2)), 0.0405):
        preset += "backproject_standard"
    else:
        preset += "backproject_clinical"
    if CPU:
        preset += "_CPU"

    # Get axes:
    axes = ""
    for i in range(3):
        if not np.all(np.isclose(data.get_scan_geometry()[:, i],  data.get_scan_geometry()[0, i])):
            axes += "xyz"[i]

    preset += "_" + axes
    return os.path.join(root_folder, preset + ".json")


def get_default_unmixing_preset():
    """

    Returns
    -------

    """
    import os
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../unmixing/unmix_presets")
    return os.path.join(root_folder, "haemoglobin.json")
