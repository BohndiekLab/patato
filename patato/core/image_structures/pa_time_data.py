"""
pa_time_data. Defines the time-domain version of PARawData class.
"""

#  Copyright (c) Thomas Else 2023.
#  License: MIT


import numpy as np
import xarray

from ..image_structures.pa_raw_data import PARawData


try:
    from pyopencl.array import Array, to_device
except ImportError:
    Array = np.ndarray
    to_device = None


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
