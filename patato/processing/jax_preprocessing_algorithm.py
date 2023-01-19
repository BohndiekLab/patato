#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from functools import partial
from typing import Union, Tuple, Optional, Dict

import numpy as np
from patato.io.attribute_tags import PreprocessingAttributeTags
from scipy.signal.windows import hann

from .processing_algorithm import TimeSeriesProcessingAlgorithm

import jax.numpy as jnp
import jax

from ..core.image_structures.pa_time_data import PATimeSeries

# Specify Array type
Array = np.typing.NDArray

@jax.jit
def subtract_mean(time_series):
    # A function to subtract the mean from a time series
    return time_series - jnp.mean(time_series, axis=-1).reshape(time_series.shape[:-1] + (1,))


@partial(jax.jit, static_argnums=(1,))
def interpolate_detectors(detectors, ndet):
    # Interpolate the detectors to the correct number of detectors
    new_detector_i = jnp.linspace(0, detectors.shape[0] - 1, ndet * detectors.shape[0])
    old_detector_i = jnp.arange(detectors.shape[0])
    interp_detectors = jax.vmap(jnp.interp, in_axes=(None, None, -1), out_axes=-1)
    new_detectors = interp_detectors(new_detector_i, old_detector_i, detectors)
    return new_detectors


@partial(jax.jit, static_argnums=(1, 2))
def partial_interpolate(time_series: Array, nt: int, ndet: int) -> Array:
    # Interpolate the time series to the correct number of time points
    new_detector_i = jnp.linspace(0, time_series.shape[-2] - 1, ndet * time_series.shape[-2])
    old_detector_i = jnp.arange(time_series.shape[-2])
    new_times = jnp.linspace(0, time_series.shape[-1] - 1, nt * time_series.shape[-1] + 1)[:-1]
    old_times = jnp.arange(time_series.shape[-1])

    interp_detectors = jax.vmap(jnp.interp, in_axes=(None, None, -1), out_axes=-1)
    new_time_series = interp_detectors(new_detector_i, old_detector_i, time_series)

    interp_time = jax.vmap(jnp.interp, in_axes=(None, None, 0), out_axes=0)
    new_time_series = interp_time(new_times, old_times, new_time_series)
    return new_time_series


def make_filter(n_samples: int, fs: float, irf: Array, hilbert: bool, lp_filter: Optional[float],
                hp_filter: Optional[float], rise: float = 0.2, n_filter: int = 1024, window: Optional[str] = None):
    """
    Make the filter for the time series

    Parameters
    ----------
    n_samples : int
    fs : float
    irf : Array
    hilbert : bool
    lp_filter: float or None
    hp_filter: float or None
    rise: float
    n_filter : int
    window

    Returns
    -------

    """
    output = np.ones((n_samples,), dtype=np.cdouble)

    # Impulse response correction
    if irf is not None:
        irf_shifted = np.fft.fftshift(irf)
        # Divide by the impulse response to deconvolve
        output *= np.conj(np.fft.fft(irf_shifted)) / np.abs(np.fft.fft(irf_shifted)) ** 2
        # Suppress high frequencies to avoid amplifying noise - apply a window.
        output *= np.fft.fftshift(hann(n_samples))

    # Hilbert Transform
    frequencies = np.fft.fftfreq(n_samples)
    if hilbert:
        # TODO: check this
        # Multiply positive frequencies by
        output *= (1 + np.sign(frequencies)) / 2

    frequencies = np.abs(np.fft.fftfreq(n_filter, 1 / fs))
    filter_output = np.ones_like(frequencies, dtype=np.cdouble)

    if hp_filter is not None:
        filter_output[frequencies < hp_filter * (1 - rise)] = 0
        in_rise = np.logical_and(frequencies > hp_filter * (1 - rise), frequencies < hp_filter)
        filter_output[in_rise] = (frequencies[in_rise] - hp_filter * (1 - rise)) / (hp_filter * rise)

    if lp_filter is not None:
        filter_output[frequencies > lp_filter * (1 + rise)] = 0
        in_rise = np.logical_and(frequencies < lp_filter * (1 + rise), frequencies > lp_filter)
        filter_output[in_rise] = 1 - (frequencies[in_rise] - lp_filter) / (lp_filter * rise)

    time_series = np.fft.ifft(filter_output)

    if window == "hann":
        time_series *= np.fft.fftshift(hann(n_filter))

    filter_time = np.zeros_like(output)
    filter_time[:n_filter // 2] = time_series[:n_filter // 2]
    filter_time[-n_filter // 2:] = time_series[-n_filter // 2:]

    filter_output = np.fft.fft(filter_time)
    output *= filter_output
    return output


class MSOTPreProcessor(TimeSeriesProcessingAlgorithm):
    """ Preprocesses MSOT time series data. Uses JAX in the background.
    """

    @staticmethod
    def get_algorithm_name() -> Union[str, None]:
        """
        Get the name of the algorithm.

        Returns
        -------
        str or None
        """
        # Return the name of the algorithm
        return "Standard Preprocessor"

    @staticmethod
    def get_hdf5_group_name() -> Union[str, None]:
        """
        Return the name of the group in the HDF5 file

        Returns
        -------
        str or None

        """
        # Return the name of the group in the HDF5 file
        return None

    def __init__(self, time_factor: int = 3, detector_factor: int = 2,
                 irf: bool = True, hilbert: bool = True, lp_filter: Optional[float] = None,
                 hp_filter: Optional[float] = None, filter_window_size: int = 512,
                 window: str = "hann", absolute: Optional[str] = None, universal_backprojection=False):
        # Initialise the preprocessor
        super().__init__()
        self.time_factor = time_factor
        self.detector_factor = detector_factor
        self.hilbert = hilbert
        absolute = "imag" if absolute is None and hilbert else absolute
        self.ubp = universal_backprojection
        self.irf_correct = irf
        self.lp_filter = lp_filter
        self.hp_filter = hp_filter
        self.n_filter = filter_window_size
        self.window = window
        self.absolute = absolute
        self.filter = None

    def pre_compute_filter(self, n_samples: int, fs: float, irf: Array = None):
        """
        Precompute the filter to be applied.

        Parameters
        ----------
        n_samples : int
        fs : float
        irf : Array
        """
        self.filter = jnp.array(make_filter(n_samples, fs, irf, self.hilbert, self.lp_filter, self.hp_filter))

    def _run(self, time_series: Array, detectors: Array, **kwargs):
        """
        Run the preprocessing step on a given time series and detectors. This allows batch processing,
         e.g. if the data doesn't fit into memory.

        Parameters
        ----------
        time_series : Array
        detectors   : Array
        kwargs    : dict

        Returns
        -------
        tuple of Array, Array
        """
        shape = time_series.shape
        time_series = jnp.array(time_series.reshape((-1,) + shape[-2:]))
        detectors = jnp.array(detectors)

        time_series = subtract_mean(time_series)
        time_series_ft = jnp.fft.fft(time_series)

        time_series_ft = time_series_ft * self.filter.reshape((1,) * (time_series.ndim - 1) + (-1,))
        if self.absolute == "imag":
            op = jnp.imag
        else:
            op = jnp.real
        time_series = op(jnp.fft.ifft(time_series_ft))

        # Allow for universal backprojection here.
        if self.ubp:
            time_series -= jnp.gradient(time_series, axis=-1)*jnp.arange(time_series.shape[-1])

        if not (self.detector_factor == 1 and self.time_factor == 1):
            full_interpolate = jax.vmap(partial_interpolate, in_axes=(0, None, None), out_axes=0)
            time_series = full_interpolate(time_series,
                                           self.time_factor,
                                           self.detector_factor)
            detectors = interpolate_detectors(detectors, self.detector_factor)
        time_series = time_series.reshape(shape[:-2] + time_series.shape[1:])
        return time_series, detectors

    def run(self, time_series, pa_data=None, irf=None, detectors=None, **kwargs) -> Tuple[
        PATimeSeries, Dict, Optional[list]]:
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

        # Sampling frequency
        fs = time_series.attributes["fs"]

        if self.filter is None:
            self.pre_compute_filter(time_series.shape[-1], fs=fs, irf=irf)

        new_detectors = detectors
        if time_series.shape[0] * time_series.shape[1] > PAT_MAXIMUM_BATCH_SIZE != -1:
            new_timeseries = []
            ts_raw = time_series.raw_data
            shape = ts_raw.shape
            ts_raw = ts_raw.reshape((-1,) + shape[-2:])
            for i in range(0, ts_raw.shape[0], PAT_MAXIMUM_BATCH_SIZE):
                new_ts, new_detectors = self._run(ts_raw[i:i + PAT_MAXIMUM_BATCH_SIZE], detectors)
                new_timeseries.append(np.asarray(new_ts))
            new_ts = np.concatenate(new_timeseries, axis=0).reshape(shape[:2] + new_timeseries[0].shape[-2:])
        else:
            new_ts, new_detectors = self._run(time_series.raw_data, detectors)

        # Convert timeseries into an xarray
        attributes = dict(time_series.attributes)
        attributes["fs"] *= self.time_factor
        attributes[PreprocessingAttributeTags.IMPULSE_RESPONSE] = self.irf_correct
        attributes[PreprocessingAttributeTags.PROCESSING_ALGORITHM] = self.get_algorithm_name()
        attributes[PreprocessingAttributeTags.WINDOW_SIZE] = self.window
        attributes[PreprocessingAttributeTags.ENVELOPE_DETECTION] = self.absolute == "abs"
        attributes[PreprocessingAttributeTags.HILBERT_TRANSFORM] = self.hilbert
        attributes[PreprocessingAttributeTags.DETECTOR_INTERPOLATION] = self.detector_factor
        attributes[PreprocessingAttributeTags.TIME_INTERPOLATION] = self.time_factor
        attributes[PreprocessingAttributeTags.LOW_PASS_FILTER] = self.lp_filter
        attributes[PreprocessingAttributeTags.HIGH_PASS_FILTER] = self.hp_filter
        attributes["UniversalBackProjection"] = self.ubp

        coords = dict(time_series.da.coords)
        coords["detectors"] = np.linspace(0, time_series.shape[-2] - 1,
                                          self.detector_factor * time_series.shape[-2])
        coords["timeseries"] = np.linspace(0, time_series.shape[-1] - 1,
                                           self.time_factor * time_series.shape[-1] + 1)[:-1]

        new_data = PATimeSeries(new_ts, time_series.da.dims, coords, attributes=attributes)
        return new_data, {"geometry": new_detectors}, None
