"""
Photoacoustic fourier data. This implements a fourier-transformed PA data type.
"""

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from __future__ import annotations

import numpy as np
import scipy.fft
import xarray
from ..image_structures.pa_raw_data import PARawData
from ..image_structures.pa_time_data import PATimeSeries


class PAFourierDomain(PARawData):
    """
    PAFourierDomain implements a datatype for fourier transforms of raw data.
    """
    def get_hdf5_group_name(self):
        return None

    def two_dims(self):
        return "detectors", "frequency_timeseries"

    def to_time_domain(self, from_complex=np.imag) -> "PATimeSeries":
        res = self.copy(PATimeSeries)
        try:
            import cupy as cp
        except ImportError:
            cp = None
        if type(res.raw_data) == xarray.DataArray:
            res.raw_data = from_complex(xr_ifft(self.raw_data, axis=-1))
        elif cp is not None and type(res.raw_data) == cp.ndarray:
            res.raw_data = from_complex(cp.fft.ifft(res.raw_data, axis=-1))
        else:
            res.raw_data = from_complex(scipy.fft.ifft(self.raw_data, axis=-1))
        return res

    def to_fourier_domain(self):
        return self
