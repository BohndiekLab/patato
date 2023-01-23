#  Copyright (c) Thomas Else 2023.
#  License: MIT

from __future__ import annotations

from typing import Union, TYPE_CHECKING, Tuple, Optional

from .preprocessing_algorithm import DefaultMSOTPreProcessor
from .processing_algorithm import ProcessingResult

if TYPE_CHECKING:
    from ..io.msot_data import PAData
    from ..core.image_structures.pa_raw_data import PARawData

from ..core.image_structures.pa_time_data import PATimeSeries

try:
    import cupy as cp
    from cupy.fft import fft, ifft, fftshift, fftfreq
except ImportError:
    cp = None

from ..io.attribute_tags import PreprocessingAttributeTags
import time
from patato.unmixing.spectra import SPECTRA_NAMES
import logging


class GPUMSOTPreProcessor(DefaultMSOTPreProcessor):
    @staticmethod
    def get_algorithm_name() -> str:
        return "GPU Standard Preprocessor"

    @staticmethod
    def get_hdf5_group_name() -> Union[str, None]:
        return None

    def __init__(self, time_factor=3, detector_factor=2,
                 irf=True, hilbert=True, lp_filter=None,
                 hp_filter=None, filter_window_size=512,
                 window: Union[str, None] = "hann", absolute: Union[bool, str] = None,
                 couplant_correction=None, couplant_path_length=0):
        super().__init__()
        self.time_factor = time_factor
        self.detector_factor = detector_factor
        absolute = "imag" if absolute is None and hilbert else absolute
        self.irf_correct = irf
        self.hilbert = hilbert
        self.lp_filter = lp_filter
        self.hp_filter = hp_filter
        self.n_filter = filter_window_size
        self.window = window
        self.absolute = absolute

        if couplant_correction is not None:
            self.couplant_correction = SPECTRA_NAMES[couplant_correction]
        else:
            self.couplant_correction = None
        self.couplant_path_length = couplant_path_length

    def run(self, time_series: PATimeSeries,
            pa_data: PAData, irf=None, detectors=None, **kwargs) -> Tuple["PATimeSeries", dict,
                                                                          Optional[ProcessingResult]]:
        if irf is None:
            irf = pa_data.get_impulse_response()
        if detectors is None:
            detectors = pa_data.get_scan_geometry()
        fs = time_series.attributes["fs"]

        if pa_data.get_overall_correction_factor() is not None:
            overall_correction_factor = cp.array(pa_data.get_overall_correction_factor())

        # Generate the filter
        ft_filter = self.make_filter(time_series.shape[-1],
                                     fs,
                                     cp.array(irf) if self.irf_correct else None,
                                     self.hilbert, self.lp_filter,
                                     self.hp_filter,
                                     n_filter=self.n_filter,
                                     window=self.window)

        new_time_series, new_parameters = self._run(time_series,
                                                    ft_filter, cp.array(detectors),
                                                    cp.array(overall_correction_factor), **kwargs)

        # Update the results' attributes.
        for a in time_series.attributes:
            if a not in new_time_series.attributes:
                new_time_series.attributes[a] = time_series.attributes[a]

        new_time_series.attributes[PreprocessingAttributeTags.IMPULSE_RESPONSE] = self.irf_correct
        new_time_series.attributes[PreprocessingAttributeTags.PROCESSING_ALGORITHM] = self.get_algorithm_name()
        new_time_series.attributes[PreprocessingAttributeTags.WINDOW_SIZE] = self.window
        new_time_series.attributes[PreprocessingAttributeTags.ENVELOPE_DETECTION] = self.absolute == "abs"
        new_time_series.attributes[PreprocessingAttributeTags.HILBERT_TRANSFORM] = self.hilbert
        new_time_series.attributes[PreprocessingAttributeTags.DETECTOR_INTERPOLATION] = self.detector_factor
        new_time_series.attributes[PreprocessingAttributeTags.TIME_INTERPOLATION] = self.time_factor
        new_time_series.attributes[PreprocessingAttributeTags.LOW_PASS_FILTER] = self.lp_filter
        new_time_series.attributes[PreprocessingAttributeTags.HIGH_PASS_FILTER] = self.hp_filter

        # TODO: replace pa_data with new attributes for further processing steps.

        return new_time_series, new_parameters, None

    def _run(self, time_series: PATimeSeries, ft_filter, detectors, overall_correction_factor, **kwargs) -> Tuple[
        PATimeSeries, dict]:
        new_parameters = {}

        # Subtract mean
        extend = (slice(None, None),) * (time_series.raw_data.ndim - 1) + (None,)

        # Subtract mean
        t = time.time()
        raw_data = cp.array(time_series.raw_data)
        time_series.raw_data = raw_data - cp.mean(raw_data, axis=-1)[extend]
        logging.debug(f"Mean subtraction took {time.time() - t}s")
        # Apply a fourier domain filter.

        t = time.time()
        time_series = time_series.to_fourier_domain()
        time_series = self.apply_filter(time_series, ft_filter=ft_filter, absolute=self.absolute)

        # Go back to the time domain.
        time_series = time_series.to_time_domain()
        logging.debug(f"Filter took {time.time() - t}s")

        # Apply interpolation in time and detector domains.
        time_series, interp_params = self.interpolate(time_series, detectors)
        new_parameters.update(interp_params)

        # Apply energy correction factor
        extend = (slice(None, None),) * overall_correction_factor.ndim + (None, None)
        time_series.raw_data /= overall_correction_factor[extend]

        # Apply the couplant correction.
        if self.couplant_correction is not None:
            # TODO: This needs rewriting.
            raise NotImplementedError("Couplant correction is not implemented.")
            # extend = (None,) * (time_series.ndim - 3) + (slice(None, None), None, None)
            # time_series.raw_data *= \
            #     np.exp(self.couplant_path_length * self.couplant_correction.get_spectrum(pa_data.get_wavelengths()))[
            #         extend]

        time_series.raw_data = time_series.raw_data.get()
        return time_series, new_parameters

    def interpolate(self, time_series: PATimeSeries, detectors, exact_ratios=True) -> Tuple[PATimeSeries, dict]:
        # Interpolate the data in the time and detector domains.
        # Interpolate in the detector domain
        if self.time_factor == 1 and self.detector_factor == 1:
            return time_series, {"geometry": detectors}

        signal = time_series.raw_data

        # Get the new indices on which to interpolate.
        detector_ind = cp.arange(detectors.shape[0])
        new_detector_ind = cp.arange((detectors.shape[0] - 1) * self.detector_factor + 1) / self.detector_factor
        if exact_ratios:
            new_detector_ind = cp.linspace(0, detectors.shape[0] - 1, self.detector_factor * detectors.shape[0])

        # Interpolate in the sample domain
        sample_ind = cp.arange(signal.shape[-1])
        new_samp_ind = cp.arange((signal.shape[-1] - 1) * self.time_factor + 1) / self.time_factor
        if exact_ratios:
            new_samp_ind = cp.linspace(0, signal.shape[-1] - 1, self.time_factor * signal.shape[-1])

        new_signal = cp.zeros(signal.shape[:-2] + (new_detector_ind.shape[0], new_samp_ind.shape[0]),
                              dtype=signal.dtype)
        t = time.time()

        sind, dind = cp.meshgrid(new_samp_ind, new_detector_ind)

        # print(dind.shape, signal.shape)
        new_signal += signal[..., dind.astype(cp.int32), sind.astype(cp.int32)] * (1 - (dind-cp.floor(dind))) * (1 - (sind-cp.floor(sind)))
        new_signal += signal[..., dind.astype(cp.int32) + 1, sind.astype(cp.int32)] * (dind-cp.floor(dind)) * (1 - (sind-cp.floor(sind)))
        new_signal += signal[..., dind.astype(cp.int32), sind.astype(cp.int32) + 1] * (1 - (dind-cp.floor(dind))) * (sind-cp.floor(sind))
        new_signal += signal[..., dind.astype(cp.int32) + 1, sind.astype(cp.int32) + 1] * (dind-cp.floor(dind)) * (sind-cp.floor(sind))
        #
        # assert 1==2

        detectors = cp.apply_along_axis(lambda x: cp.interp(new_detector_ind, detector_ind, x),
                                        0, detectors)

        logging.debug(f"Interpolation took {time.time() - t}.")

        # Update the xarray coordinates of the new dataset.
        coords = dict(time_series.da.coords)
        coords["detectors"] = new_detector_ind.get()
        coords["timeseries"] = new_samp_ind.get()

        attributes = dict(time_series.da.attrs)
        attributes["fs"] *= self.time_factor

        new_data = PATimeSeries(new_signal, time_series.da.dims, coords, attributes=attributes)
        return new_data, {"geometry": detectors.get()}

    @staticmethod
    def apply_filter(pa_data: PARawData, ft_filter, absolute=False) -> PATimeSeries:
        pa_fft = pa_data.to_fourier_domain()
        extend = (None,) * (pa_fft.raw_data.ndim - 1) + (slice(None, None),)
        pa_fft.raw_data *= ft_filter[extend]
        operation = cp.real if absolute == "real" or absolute is None else cp.imag if absolute == "imag" else cp.abs
        pa_time = pa_fft.to_time_domain(operation)
        return pa_time

    @staticmethod
    def make_filter(n_samples, fs, irf,
                    hilbert, lp_filter,
                    hp_filter, rise=0.2,
                    n_filter=1024, window=None) -> cp.ndarray:
        # at the moment, it looks like it is shifting the data a bit??
        # Impulse Response Correction
        output = cp.ones((n_samples,), dtype=cp.cdouble)
        if irf is not None:
            irf_shifted = cp.zeros_like(irf)
            irf_shifted[:irf.shape[0] // 2] = irf[irf.shape[0] // 2:]
            irf_shifted[-irf.shape[0] // 2:] = irf[:irf.shape[0] // 2]
            output *= cp.conj(fft(irf_shifted)) / cp.abs(fft(irf_shifted)) ** 2
            from scipy.signal.windows import hann
            output *= fftshift(hann(n_samples))

        # Hilbert Transform
        frequencies = fftfreq(n_samples)
        if hilbert:
            output *= (1 + cp.sign(frequencies)) / 2

        frequencies = cp.abs(fftfreq(n_filter, 1 / fs))
        fir_filter = cp.ones_like(frequencies, dtype=cp.cdouble)
        if hp_filter is not None:
            fir_filter[frequencies < hp_filter * (1 - rise)] = 0
            in_rise = cp.logical_and(frequencies > hp_filter * (1 - rise), frequencies < hp_filter)
            fir_filter[in_rise] = (frequencies[in_rise] - hp_filter * (1 - rise)) / (hp_filter * rise)
        if lp_filter is not None:
            fir_filter[frequencies > lp_filter * (1 + rise)] = 0
            in_rise = cp.logical_and(frequencies < lp_filter * (1 + rise), frequencies > lp_filter)
            fir_filter[in_rise] = 1 - (frequencies[in_rise] - lp_filter) / (lp_filter * rise)

        time_series = ifft(fir_filter)

        if window == "hann":
            from scipy.signal.windows import hann
            time_series *= fftshift(hann(n_filter))

        filter_time = cp.zeros_like(output)
        filter_time[:n_filter // 2] = time_series[:n_filter // 2]
        filter_time[-n_filter // 2:] = time_series[-n_filter // 2:]
        fir_filter = fft(filter_time)
        output *= fir_filter
        return output
