"""
PA raw data implements the data structure for photoacoustic time series data.
"""

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ...core.image_structures.image_sequence import DataSequence
from ...io.attribute_tags import HDF5Tags

if TYPE_CHECKING:
    from ...core.image_structures.pa_fourier_data import PAFourierDomain
    from ...core.image_structures.pa_time_data import PATimeSeries


class PARawData(DataSequence, ABC):
    """
    PARawData is the abstract data structure for time series data. This can be in the pure time series format or the
    fourier way.
    """

    def __add__(self, other):
        raise NotImplementedError()
    n_im_dim = 2

    def __init__(self, data, dimensions, coordinates=None, attributes=None, hdf5_sub_name=None,
                 algorithm_id=None):
        """

        Parameters
        ----------
        data
        dimensions
        coordinates
        attributes
        hdf5_sub_name
        algorithm_id
        """
        super().__init__(data, dimensions, coordinates, attributes,
                         hdf5_sub_name=hdf5_sub_name, algorithm_id=algorithm_id)
        self.save_output = False

    @staticmethod
    def is_single_instance():
        """

        Returns
        -------

        """
        return True

    @staticmethod
    def get_ax1_label_meaning():
        """

        Returns
        -------

        """
        return HDF5Tags.WAVELENGTH

    @abstractmethod
    def to_time_domain(self, from_complex=np.imag) -> "PATimeSeries":
        """

        Parameters
        ----------
        from_complex

        Returns
        -----------
        PATimeSeries
        """
        pass

    @abstractmethod
    def to_fourier_domain(self) -> "PAFourierDomain":
        """
        Returns
        -------
        PAFourierDomain
        """
        pass
