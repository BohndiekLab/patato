"""
Defines a datatype for datasets that have been processed to only have one value per scan per pixel.
"""

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import h5py

from ...core.image_structures.image_sequence import ImageSequence


class SingleImage(ImageSequence):
    """
    SingleImage is the datastructure for images like delta sO2..
    """

    @staticmethod
    def get_ax1_label_meaning():
        return None
    save_output = True

    def get_hdf5_group_name(self):
        return self.ax_1_labels.item()

    @staticmethod
    def get_ax1_labels_from_hdf5(dataset: "h5py.Dataset", file: "h5py.File") -> List[str]:
        return [dataset.name.split("/")[-3]]

    @staticmethod
    def is_single_instance():
        return False

    @property
    def ax_0_labels(self):
        return None

    @staticmethod
    def ax_0_exists():
        return False
