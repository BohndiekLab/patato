#  Copyright (c) Thomas Else 2023.
#  Copyright (c) Janek Grohl 2023.
#  License: MIT

from os.path import split
import pacfish as pf
import numpy as np
from ..hdf.fileimporter import ReaderInterface


class IPASCInterface(ReaderInterface):
    """
    An interface for iThera MSOT datasets.
    """
    def _get_rois(self):
        pass

    def _get_segmentation(self):
        return None

    def _get_datasets(self):
        return None

    def get_speed_of_sound(self):
        return None

    def __init__(self, file_path):
        super().__init__()
        self.scan_name = split(file_path)[-1]
        self.pa_data = pf.load_data(file_path)
        self.nwavelengths = len(self.pa_data.get_acquisition_wavelengths())
        self.nframes = np.shape(self.pa_data.binary_time_series_data)[3]
        # Optional add here: extract the reconstructed images that are from ViewMSOT.
        # Extract attributes
        self.geometry = self.pa_data.get_detector_position()
        self.nsamples = self.nwavelengths * self.nframes

    def get_n_samples(self):
        return self.nsamples

    def _get_wavelengths(self):
        return self.pa_data.get_acquisition_wavelengths()

    def _get_correction_factor(self):
        return self.pa_data.get_overall_gain()

    def get_impulse_response(self):
        return self.pa_data.get_frequency_response()

    def _get_repetition_numbers(self):
        return self.nsamples

    def _get_run_numbers(self):
        return np.arange(1, self.nframes+1)

    def get_scan_datetime(self):
        return None

    def _get_scan_times(self):
        return self.pa_data.get_measurement_time_stamps()

    def _get_temperature(self):
        return self.pa_data.get_temperature()

    def get_us_offsets(self):
        return None

    def _get_pa_data(self):
        raw_data = self.pa_data.binary_time_series_data # [detectors, samples, wavelengths, frames]
        raw_data = np.swapaxes(raw_data, 1, 3)  # [detectors, frames, wavelengths, samples]
        raw_data = np.swapaxes(raw_data, 0, 2)  # [wavelengths, frames, detectors, samples]
        raw_data = np.swapaxes(raw_data, 0, 1)  # [frames, wavelengths, detectors, samples]
        return raw_data, {"fs": self.pa_data.get_sampling_rate()}

    def _get_sampling_frequency(self):
        return self.pa_data.get_sampling_rate()

    def _get_sensor_geometry(self):
        return self.pa_data.get_detector_position()

    def _get_water_absorption(self):
        return None

    def get_us_data(self):
        return None

    def get_scan_name(self):
        return self.scan_name

    def _get_scanner_z_position(self):
        return self.pa_data.get_measurement_spatial_poses()

    def get_scan_comment(self):
        return ""
