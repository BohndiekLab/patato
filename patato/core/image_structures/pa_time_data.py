"""
pa_time_data. Defines the time-domain version of PARawData class.
"""

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from typing import TYPE_CHECKING

import numpy as np
import scipy.fft
import xarray

from ..image_structures.pa_raw_data import PARawData

if TYPE_CHECKING:
    from ..image_structures.pa_fourier_data import PAFourierDomain

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

    def to_fourier_domain(self) -> "PAFourierDomain":
        from ...core.image_structures.pa_fourier_data import PAFourierDomain
        res = self.copy(PAFourierDomain)
        try:
            import cupy as cp
        except ImportError:
            cp = None
        if cp is not None and type(self.raw_data) == cp.ndarray:
            res.raw_data = cp.fft.fft(self.raw_data, axis=-1)
        else:
            res.raw_data = scipy.fft.fft(self.raw_data, axis=-1)
        return res

    def to_time_domain(self, _=None) -> "PATimeSeries":
        return self
