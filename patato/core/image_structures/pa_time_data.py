"""
pa_time_data. Defines the time-domain version of PARawData class.
"""
#  Copyright (c) Thomas Else 2023.
#  License: MIT


import numpy as np
import xarray
from typing import Optional
import numpy.typing as npt

from ..image_structures.pa_raw_data import PARawData


class PATimeSeries(PARawData):
    """
    PATimeSeries is the data structure for time-domain raw PA data.
    """

    def __add__(self, other):
        new_data = xarray.concat([self.da, other.da], dim=other.da.dims[0])
        output = PATimeSeries(new_data, new_data.dims, new_data.coords, new_data.attrs,
                              self.hdf5_sub_name, self.algorithm_id)
        output.__class__ = self.__class__
        return output

    def get_hdf5_group_name(self):
        return "raw_data"

    def to_opencl(self, queue) -> "PATimeSeries":
        # Only import if needed.
        from pyopencl.array import to_device
        cls = self.copy()
        cls.da = cls.da.copy()
        new_data = to_device(queue, self.raw_data.astype(np.single))

        # Fudge to make pyopencl ducktyped with numpy array
        new_data.__array_function__ = None
        new_data.__array_ufunc__ = None

        cls.raw_data = new_data
        return cls

    def two_dims(self):
        return "detectors", "timeseries"

    @classmethod
    def from_numpy(cls, dataset: npt.ArrayLike, wavelengths: npt.ArrayLike, fs: float,
                   speed_of_sound: Optional[float] = None):
        """
        Create a PATimeSeries class from a NumPy array.

        Parameters
        ----------
        dataset: np.ndarray
        wavelengths: np.ndarray
        fs: float
        speed_of_sound: float

        Returns
        -------
        PATimeSeries
        """
        dims = ["frames", cls.get_ax1_label_meaning(), "detectors", "timeseries"]
        dim_coords = [np.arange(dataset.shape[0]), wavelengths,
                      np.arange(dataset.shape[2]), np.arange(dataset.shape[3])
                      ]
        coordinates = {a: b for a, b in zip(dims, dim_coords)}
        attributes = {'fs': fs, 'speedofsound': speed_of_sound}
        return cls(dataset, dims, coordinates, attributes)
