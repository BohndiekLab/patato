#  Copyright (c) Thomas Else 2024.
#  License: MIT

import numpy as np
from patato.io.attribute_tags import PreprocessingAttributeTags
from scipy.signal import butter, filtfilt

from .processing_algorithm import TimeSeriesProcessingAlgorithm

import warnings

from ..core.image_structures.pa_time_data import PATimeSeries
from typing import Dict, Optional, Tuple, Union


class NumpyBasicPreProcessor(TimeSeriesProcessingAlgorithm):
    """Preprocesses MSOT time series data. Uses JAX in the background."""

    @staticmethod
    def get_algorithm_name() -> Union[str, None]:
        """
        Get the name of the algorithm.

        Returns
        -------
        str or None
        """
        return "Butterworth Filter Preprocessor"

    @staticmethod
    def get_hdf5_group_name() -> Union[str, None]:
        """
        Return the name of the group in the HDF5 file.

        Returns
        -------
        str or None
        """
        # Return the name of the group in the HDF5 file
        return None

    def __init__(
        self,
        lp_filter: Optional[float] = None,
        hp_filter: Optional[float] = None,
    ):
        # Initialise the preprocessor
        super().__init__()
        self.lp_filter = lp_filter
        self.hp_filter = hp_filter

        self.filter = None

    def _run(
        self,
        time_series: np.ndarray,
        _: np.ndarray = None,
        overall_correction_factor: np.ndarray = None,
        **kwargs
    ):
        """
        Run the preprocessing step on a given time series and detectors. This allows batch processing,
         e.g. if the data doesn't fit into memory.

        Parameters
        ----------
        time_series : Array
        detectors : Array
        overall_correction_factor : Array
        kwargs : dict

        Returns
        -------
        tuple of Array, Array
        """
        shape = time_series.shape
        time_series = time_series.reshape((-1,) + shape[-2:])

        # TODO: Do the filtering here.
        b, a = self.filter
        time_series = filtfilt(
            b, a, time_series, axis=-1, padtype="even", padlen=100, method="pad"
        )
        time_series = time_series.reshape(shape[:-2] + time_series.shape[1:])

        # Apply energy correction factor:
        if overall_correction_factor is not None:
            extend = (slice(None, None),) * overall_correction_factor.ndim + (
                None,
                None,
            )
            time_series /= overall_correction_factor[extend]
        else:
            warnings.warn("No energy correction factor applied.")

        return time_series, None

    def run(
        self, time_series, pa_data=None, irf=None, detectors=None, **kwargs
    ) -> Tuple[PATimeSeries, Dict, Optional[list]]:
        """
        Run the preprocessing step on a given time series and detectors. This allows batch processing,
        e.g. if the data doesn't fit into memory.

        Parameters
        ----------
        time_series
        pa_data
        irf
        detectors
        kwargs

        Returns
        -------
        tuple of PATimeSeries, dict, list
        """
        from .. import PAT_MAXIMUM_BATCH_SIZE

        # Impulse response
        if irf is None and pa_data is not None:
            irf = pa_data.get_impulse_response()

        # Photoacoustic transducers
        if detectors is None and pa_data is not None:
            detectors = pa_data.get_scan_geometry()

        if pa_data is not None:
            overall_correction_factor = pa_data.get_overall_correction_factor()
        else:
            overall_correction_factor = None

        # Sampling frequency
        fs = time_series.attributes["fs"]

        self.filter = butter(
            5, [self.hp_filter, self.lp_filter], btype="bandpass", fs=fs
        )

        if time_series.shape[0] * time_series.shape[1] > PAT_MAXIMUM_BATCH_SIZE != -1:
            new_timeseries = []
            ts_raw = time_series.raw_data
            shape = ts_raw.shape
            ts_raw = ts_raw.reshape((-1,) + shape[-2:])
            for i in range(0, ts_raw.shape[0], PAT_MAXIMUM_BATCH_SIZE):
                if overall_correction_factor is not None:
                    overall_correction_factor_sliced = (
                        overall_correction_factor.flatten()[
                            i : i + PAT_MAXIMUM_BATCH_SIZE
                        ]
                    )
                else:
                    overall_correction_factor_sliced = None
                new_ts, new_detectors = self._run(
                    ts_raw[i : i + PAT_MAXIMUM_BATCH_SIZE],
                    detectors,
                    overall_correction_factor_sliced,
                )
                new_timeseries.append(np.asarray(new_ts))
            new_ts = np.concatenate(new_timeseries, axis=0).reshape(
                shape[:2] + new_timeseries[0].shape[-2:]
            )
        else:
            new_ts, new_detectors = self._run(
                time_series.raw_data, detectors, overall_correction_factor
            )

        # Convert timeseries into an xarray
        attributes = dict(time_series.attributes)
        attributes[
            PreprocessingAttributeTags.PROCESSING_ALGORITHM
        ] = self.get_algorithm_name()
        attributes[PreprocessingAttributeTags.LOW_PASS_FILTER] = self.lp_filter
        attributes[PreprocessingAttributeTags.HIGH_PASS_FILTER] = self.hp_filter
        attributes["CorrectionFactorApplied"] = overall_correction_factor is not None

        coords = dict(time_series.da.coords)
        coords["detectors"] = np.linspace(
            0, time_series.shape[-2] - 1, time_series.shape[-2]
        )
        coords["timeseries"] = np.linspace(
            0, time_series.shape[-1] - 1, time_series.shape[-1] + 1
        )[:-1]

        new_data = PATimeSeries(
            new_ts, time_series.da.dims, coords, attributes=attributes
        )
        return new_data, {}, None
