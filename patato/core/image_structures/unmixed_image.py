"""
This defines the data structure for unmixed datasets.
"""

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import h5py
    import numpy as np

from ...core.image_structures.image_sequence import ImageSequence
from ...io.attribute_tags import HDF5Tags


class UnmixedData(ImageSequence):
    """
    UnmixedData stores unmixed datasets.
    """

    save_output = True
    @staticmethod
    def is_single_instance():
        return False

    @staticmethod
    def get_ax1_label_meaning():
        return "SPECTRA"

    def get_hdf5_group_name(self):
        return HDF5Tags.UNMIXED

    @staticmethod
    def get_ax1_labels_from_hdf5(dataset: "h5py.Dataset", file: "h5py.File") -> Optional["np.ndarray"]:
        return dataset.attrs["SPECTRA"]

    @property
    def spectra(self):
        return self.ax_1_labels
