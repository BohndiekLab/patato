#  Copyright (c) Thomas Else 2023.
#  License: MIT

import numpy as np

try:
    import simpa as sp
except ImportError:
    sp = None

from ...core.image_structures.reconstruction_image import Reconstruction
from ..attribute_tags import ReconAttributeTags
from ..hdf.fileimporter import ReaderInterface


class SimpaImporter(ReaderInterface):
    """An importer for HDF5 files created by the SIMPA toolkit.
    """
    # Currently, aiming to just support 2D
    def _get_rois(self):
        return None

    def get_speed_of_sound(self):
        f = sp.get_data_field_from_simpa_output(self.file, sp.Tags.DATA_FIELD_SPEED_OF_SOUND)
        return np.mean(f)

    def _original_shape(self):
        return self._get_segmentation().shape

    def _get_segmentation(self):
        seg = sp.get_data_field_from_simpa_output(self.file, sp.Tags.DATA_FIELD_SEGMENTATION)
        if seg.ndim == 2:
            seg = seg[:, :, None]
        return seg

    def get_scan_comment(self):
        return ""

    def _get_sampling_frequency(self):
        if "dt_acoustic_sim" not in self.file["settings"]:
            return 1
        return 1 / self.file["settings"]["dt_acoustic_sim"]

    def _simpa_get_initial_pressure(self, wavelength):
        recon = sp.get_data_field_from_simpa_output(self.file, sp.Tags.DATA_FIELD_INITIAL_PRESSURE,
                                                    wavelength)
        if recon.ndim == 2:
            recon = recon[:, :, None]
        return recon

    def _get_datasets(self):
        wavelengths = self.wavelengths
        recon_size = self._simpa_get_initial_pressure(wavelengths[0]).shape

        recon_data = np.zeros((1, len(self.wavelengths),) + recon_size)
        
        for i, w in enumerate(wavelengths):
            recon_data[0, i] = self._simpa_get_initial_pressure(w)

        attributes = {ReconAttributeTags.X_NUMBER_OF_PIXELS: recon_data.shape[-1],
                      ReconAttributeTags.Y_NUMBER_OF_PIXELS: recon_data.shape[-2],
                      ReconAttributeTags.Z_NUMBER_OF_PIXELS: recon_data.shape[-3],
                      ReconAttributeTags.X_FIELD_OF_VIEW: self._simpa_fov[0],
                      ReconAttributeTags.Y_FIELD_OF_VIEW: self._simpa_fov[1],
                      ReconAttributeTags.Z_FIELD_OF_VIEW: self._simpa_fov[2]}
        # recon_data = recon_data.swapaxes(0, 3).copy()

        recon_class = Reconstruction(recon_data, self._get_wavelengths(),
                                     attributes=dict(attributes),
                                     field_of_view=self._simpa_fov, hdf5_sub_name="initial_pressure")

        return {"recons": {("initial_pressure", "0"): recon_class}}

    def __init__(self, filename):
        super().__init__()
        self.file = sp.load_hdf5(filename)
        self.filename = filename
        
        z0, _, x0, _, y0, _ = self.file["digital_device"].get_detection_geometry().get_field_of_view_mm()/1000
        nz, ny, nx = self._original_shape()
        dx = self.file["settings"]["voxel_spacing_mm"]/1000
        self._simpa_fov = [(x0, x0+dx*nx), (y0, y0+dx*ny), (z0, z0+dx*nz)]

        self.wavelengths = self.file["settings"]["wavelengths"]

        self.nz = 1 # for backwards compatibility

    def get_scan_datetime(self):
        return 0

    def _get_pa_data(self):
        try:
            attrs = {"fs": self.get_sampling_frequency()}
            pa = np.array([sp.get_data_field_from_simpa_output(self.file, sp.Tags.DATA_FIELD_TIME_SERIES_DATA, w) for w in
                           self.wavelengths])

            return (pa[None],
                    attrs)
        except KeyError:
            return (np.full((1, len(self.wavelengths), 256, 2), np.nan), {})

    def get_scan_name(self):
        return self.filename

    def _get_temperature(self):
        return np.zeros((self.nz, len(self.wavelengths)))

    def _get_correction_factor(self):
        return np.ones((self.nz, len(self.wavelengths)))

    def _get_scanner_z_position(self):
        return np.arange(self.nz)[:, None] * np.ones((len(self.wavelengths),))[None, :] * self.file["settings"][
            "voxel_spacing_mm"]

    def _get_run_numbers(self):
        return np.zeros((self.nz, len(self.wavelengths)))

    def _get_repetition_numbers(self):
        return np.zeros((self.nz, len(self.wavelengths)))

    def _get_scan_times(self):
        return np.arange(self.nz * len(self.wavelengths)).reshape((self.nz, len(self.wavelengths)))

    def _get_sensor_geometry(self):
        g = self.file["digital_device"].get_detection_geometry().get_detector_element_positions_accounting_for_field_of_view()
        return g / 1000

    def get_us_data(self):
        return None

    def get_impulse_response(self):
        irf = np.zeros((self.get_pa_data()[0].shape[-1],))
        irf[irf.shape[0] // 2] = 1
        return irf

    def _get_wavelengths(self):
        return self.wavelengths

    def get_us_offsets(self):
        return None

    def _get_water_absorption(self):
        return np.zeros((len(self.wavelengths),)), 0
