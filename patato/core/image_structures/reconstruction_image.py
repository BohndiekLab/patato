"""
This defines the data structure for reconstructed images.
"""

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from __future__ import annotations

import numpy as np

from ...core.image_structures.image_sequence import ImageSequence
from ...io.attribute_tags import HDF5Tags


class Reconstruction(ImageSequence):
    """Data structure for reconstructed images.
    """
    save_output = True
    @staticmethod
    def is_single_instance():
        return False

    @staticmethod
    def get_ax1_label_meaning():
        return HDF5Tags.WAVELENGTH

    def get_hdf5_group_name(self) -> str:
        return HDF5Tags.RECONSTRUCTION

    @property
    def wavelengths(self):
        return np.array(self.ax_1_labels)

    @classmethod
    def from_numpy(cls, data, wavelength):
        return cls()
