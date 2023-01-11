#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import numpy as np
import pacfish as pf
from pacfish import MetaDatum


class TomHDF5AdapterToIPASCFormat(pf.BaseAdapter):
    def generate_device_meta_data(self) -> dict:
        return {}

    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        super(TomHDF5AdapterToIPASCFormat, self).__init__()

    def generate_binary_data(self) -> np.ndarray:
        # iThera definition: [frames, wavelengths, detectors, time_series]
        # IPASC definition: [detectors, time_series, wavelength, frames]
        time_series = self.hdf5_file["raw_data"]
        time_series = np.swapaxes(time_series, 2, 0)  # [detectors, wavelengths, frames, time_series]
        time_series = np.swapaxes(time_series, 3, 1)  # [detectors, time_series, frames, wavelength]
        time_series = np.swapaxes(time_series, 3, 2)  # [detectors, time_series, wavelength, frames]
        return time_series

    def generate_meta_data_device(self) -> dict:
        device_metadata_creator = pf.DeviceMetaDataCreator()
        for array_element in np.asarray(self.hdf5_file["GEOMETRY"]):
            det_element = pf.DetectionElementCreator()
            det_element.set_detector_position(array_element)
            device_metadata_creator.add_detection_element(det_element.get_dictionary())
        return device_metadata_creator.finalize_device_meta_data()

    def set_metadata_value(self, metadata_tag: MetaDatum) -> object:
        if metadata_tag == pf.MetadataAcquisitionTags.PULSE_ENERGY:
            return np.asarray(self.hdf5_file['OverallCorrectionFactor'])
        if metadata_tag == pf.MetadataAcquisitionTags.TEMPERATURE_CONTROL:
            return np.asarray(self.hdf5_file["TEMPERATURE"])
        if metadata_tag == pf.MetadataAcquisitionTags.MEASUREMENT_SPATIAL_POSES:
            # TODO convert each element into [0, POS, 0, 0, 0, 0]
            orig_shape = self.hdf5_file["Z-POS"]
            poses = []
            for y_pos in np.asarray(self.hdf5_file["Z-POS"]).reshape((-1, )):
                poses.append([0, y_pos, 0, 0, 0])

            return np.asarray(poses).reshape((orig_shape[0], orig_shape[1], -1))
        if metadata_tag == pf.MetadataAcquisitionTags.MEASUREMENT_TIMESTAMPS:
            return np.asarray(self.hdf5_file["timestamp"])
        if metadata_tag == pf.MetadataAcquisitionTags.ACQUISITION_WAVELENGTHS:
            return np.asarray(self.hdf5_file["wavelengths"])
        return None


