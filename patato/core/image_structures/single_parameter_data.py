"""
This defines the data structure for single parameter datasets like sO2 and THb.
"""

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from ..image_structures.image_sequence import ImageSequence


class SingleParameterData(ImageSequence):
    """
    SingleParameterData is the datastructure for images like sO2 and THb (one value per frame per pixel).
    """
    save_output = True
    @staticmethod
    def is_single_instance():
        return False

    @staticmethod
    def get_ax1_label_meaning():
        return "parameter"

    def get_hdf5_group_name(self):
        return self.parameters[0].item()

    @property
    def parameters(self):
        return self.ax_1_labels

    def __init__(self, raw_data, ax_1_labels=None, algorithm_id="", field_of_view=None, attributes=None,
                 hdf5_sub_name=None):
        if len(ax_1_labels) != 1 or raw_data.shape[1] != 1:
            raise ValueError("Single parameter data requires raw_data.shape[1] to be 1.")
        super().__init__(raw_data, ax_1_labels, algorithm_id, field_of_view, attributes, hdf5_sub_name)
