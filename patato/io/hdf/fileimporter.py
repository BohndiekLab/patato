#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import copy
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Union, Tuple

import dask.array as da
import h5py
import numpy as np

from ...core.image_structures.pa_time_data import PATimeSeries
from ...utils.mask_operations import interpolate_rois


def slice_1d(data, test_data, slices, dim=-1):
    def slice_wl(slice_data, item, wl_axis):
        if wl_axis == 0 and type(item) is not tuple:
            r = slice_data[item]
        elif type(item) is not tuple:
            r = slice_data
        else:
            if len(item) > wl_axis:
                r = slice_data[item[wl_axis]]
            else:
                r = slice_data
        return r

    t = data
    for s in slices:
        test_data = test_data[s]
        wl_axis = test_data.ndim + dim
        t = slice_wl(t, s, wl_axis)
    return t


class ReaderInterface(metaclass=ABCMeta):
    def close(self):
        pass

    def is_clinical(self):
        return np.all(np.isnan(self.get_scanner_z_position())) or np.all(0. == self.get_scanner_z_position())

    def save_to_hdf5(self, filename):
        from ..hdf.hdf5_interface import HDF5Writer
        file = h5py.File(filename, "a")
        writer = HDF5Writer(file)
        return writer.save_file(self)

    def __getitem__(self, item):
        s = copy.copy(self)
        s.slices = copy.deepcopy(s.slices)
        # check valid
        try:
            s._get_run_numbers()[item]
        except KeyError:
            raise KeyError("Invalid slice")
        s.slices.append(item)
        return s

    def __init__(self):
        self._raw_time_series_data = None
        self._sampling_frequency = None
        self._scan_geometry = None
        self.slices = []

    @abstractmethod
    def _get_rois(self):
        pass

    def get_rois(self, interpolate=False):
        output = self._get_rois()
        if interpolate:
            groups = defaultdict(list)
            for name, number in output.keys():
                groups[name].append(output[(name, number)])
            for roi_name in groups:
                if len(groups[roi_name]) > 0:
                    interpolated_rois = interpolate_rois(groups[roi_name], self.get_scanner_z_position())
                    for i, roi in enumerate(interpolated_rois):
                        output[(roi.roi_class + "_" + roi.position, str(i))] = roi
        return output

    @abstractmethod
    def _get_segmentation(self):
        pass

    def get_segmentation(self):
        test_data = self._get_run_numbers()
        t = self._get_segmentation()
        t = slice_1d(t, test_data, self.slices, 0)
        return t

    @abstractmethod
    def get_scan_datetime(self):
        pass

    @property
    def raw_data(self):
        return self.get_pa_data()

    @raw_data.setter
    def raw_data(self, x):
        self._raw_time_series_data = x

    @property
    def sampling_frequency(self):
        return self.get_sampling_frequency()

    @sampling_frequency.setter
    def sampling_frequency(self, x):
        self._sampling_frequency = x

    @property
    def scan_geometry(self):
        return self.get_sensor_geometry()

    @scan_geometry.setter
    def scan_geometry(self, x):
        self._scan_geometry = x

    def get_pa_data(self):
        dataset, attributes = self._get_pa_data()
        if self._raw_time_series_data is not None:
            dataset = self._raw_time_series_data
        wavelengths = self._get_wavelengths()
        cls = PATimeSeries
        dims = ["frames", cls.get_ax1_label_meaning(), "detectors", "timeseries"]

        dim_coords = [np.arange(dataset.shape[0]), wavelengths,
                      np.arange(dataset.shape[2]), np.arange(dataset.shape[3])
                      ]
        coordinates = {a: b for a, b in zip(dims, dim_coords)}
        new_cls = cls(da.from_array(dataset, chunks=(1,) + dataset.shape[1:]), dims, coordinates, attributes)

        if not cls.is_single_instance():
            new_cls.hdf5_sub_name = dataset.name.split("/")[-2]
        for s in self.slices:
            new_cls = new_cls[s]
        return new_cls

    @abstractmethod
    def _get_pa_data(self):
        pass

    @abstractmethod
    def get_scan_name(self):
        pass

    @abstractmethod
    def _get_temperature(self):
        pass

    def get_temperature(self):
        t = self._get_temperature()
        for s in self.slices:
            t = t[s]
        return t

    @abstractmethod
    def _get_correction_factor(self):
        pass

    def get_correction_factor(self):
        t = self._get_correction_factor()
        for s in self.slices:
            t = t[s]
        return t

    @abstractmethod
    def _get_scanner_z_position(self):
        pass

    def get_scanner_z_position(self):
        t = self._get_scanner_z_position()
        for s in self.slices:
            t = t[s]
        return t

    @abstractmethod
    def _get_run_numbers(self):
        pass

    def get_run_numbers(self):
        t = self._get_run_numbers()
        for s in self.slices:
            t = t[s]
        return t

    @abstractmethod
    def _get_repetition_numbers(self):
        pass

    def get_repetition_numbers(self):
        t = self._get_repetition_numbers()
        for s in self.slices:
            t = t[s]
        return t

    @abstractmethod
    def _get_scan_times(self):
        pass

    def get_scan_times(self):
        t = self._get_scan_times()
        for s in self.slices:
            t = t[s]
        return t

    @abstractmethod
    def _get_sensor_geometry(self):
        pass

    def get_sensor_geometry(self):
        if self._scan_geometry is not None:
            return self._scan_geometry
        else:
            return self._get_sensor_geometry()

    @abstractmethod
    def get_us_data(self):
        # TODO: implement slicing
        pass

    @abstractmethod
    def get_impulse_response(self):
        pass

    @abstractmethod
    def _get_wavelengths(self):
        pass

    def get_wavelengths(self):
        test_data = self._get_run_numbers()
        t = self._get_wavelengths()
        t = slice_1d(t, test_data, self.slices, -1)
        return t

    @abstractmethod
    def get_us_offsets(self):
        pass

    @abstractmethod
    def _get_water_absorption(self):
        pass

    def get_water_absorption(self):
        t, pl = self._get_water_absorption()
        test_data = self._get_run_numbers()
        t = slice_1d(t, test_data, self.slices, -1)
        return t, pl

    @abstractmethod
    def _get_datasets(self):
        # Make this return an image sequence type
        pass

    def get_datasets(self):
        all_datasets = self._get_datasets()
        if all_datasets is not None:
            for s in self.slices:
                for dataset_type in all_datasets:
                    for reconstruction_type in all_datasets[dataset_type]:
                        if all_datasets[dataset_type][reconstruction_type]:
                            all_datasets[dataset_type][reconstruction_type] = \
                                all_datasets[dataset_type][reconstruction_type][s]
        return all_datasets

    @abstractmethod
    def get_scan_comment(self):
        pass

    @abstractmethod
    def _get_sampling_frequency(self):
        pass

    def get_sampling_frequency(self):
        if self._sampling_frequency is not None:
            return self._sampling_frequency
        else:
            return self._get_sampling_frequency()

    @abstractmethod
    def get_speed_of_sound(self):
        pass


class WriterInterface(metaclass=ABCMeta):
    def close(self):
        pass

    def save_file(self, reader: ReaderInterface):
        # TODO: implement updating.
        if reader.get_segmentation() is not None:
            self.set_segmentation(reader.get_segmentation())
        self.set_scan_datetime(reader.get_scan_datetime())
        self.set_pa_data(reader.get_pa_data())
        self.set_scan_name(reader.get_scan_name())
        self.set_temperature(reader.get_temperature())
        self.set_correction_factor(reader.get_correction_factor())
        self.set_scanner_z_position(reader.get_scanner_z_position())
        self.set_run_numbers(reader.get_run_numbers())
        self.set_repetition_numbers(reader.get_repetition_numbers())
        self.set_scan_times(reader.get_scan_times())
        self.set_sensor_geometry(reader.get_sensor_geometry())
        if reader.get_us_data() is not None:
            self.set_us_data(*reader.get_us_data())  # TODO: implement us data as a image data type.
        self.set_impulse_response(reader.get_impulse_response())
        self.set_wavelengths(reader.get_wavelengths())
        self.set_us_offsets(reader.get_us_offsets())
        self.set_water_absorption(*reader.get_water_absorption())
        if reader.get_datasets() is not None:
            for _, image_group in reader.get_datasets().items():
                for key in sorted(image_group, key=lambda x: int(x[1])):
                    recon = image_group[key]
                    self.add_image(recon)
        self.set_scan_comment(reader.get_scan_comment())
        self.set_sampling_frequency(reader.get_sampling_frequency())

    @abstractmethod
    def set_segmentation(self, seg):
        pass

    @abstractmethod
    def set_scan_datetime(self, datetime):
        pass

    @abstractmethod
    def set_pa_data(self, pa_data: "PATimeSeries"):
        pass

    @abstractmethod
    def set_scan_name(self, scan_name: str):
        pass

    @abstractmethod
    def set_temperature(self, temperature: "np.ndarray"):
        pass

    @abstractmethod
    def set_correction_factor(self, correction_factor):
        pass

    @abstractmethod
    def set_scanner_z_position(self, z_position):
        pass

    @abstractmethod
    def set_run_numbers(self, run_numbers):
        pass

    @abstractmethod
    def set_repetition_numbers(self, repetition_numbers):
        pass

    @abstractmethod
    def set_scan_times(self, scan_times):
        pass

    @abstractmethod
    def set_sensor_geometry(self, sensor_geometry):
        pass

    @abstractmethod
    def set_us_data(self, us_data, us_fov):
        pass

    @abstractmethod
    def set_impulse_response(self, impulse_response):
        pass

    @abstractmethod
    def set_wavelengths(self, wavelengths):
        pass

    @abstractmethod
    def set_us_offsets(self, us_offsets):
        pass

    @abstractmethod
    def set_water_absorption(self, water_absorption, pathlength):
        pass

    @abstractmethod
    def add_image(self, image):
        pass

    @abstractmethod
    def delete_images(self, image):
        pass

    @abstractmethod
    def set_scan_comment(self, comment: str):
        pass

    @abstractmethod
    def set_sampling_frequency(self, frequency: float):
        pass

    @abstractmethod
    def set_speed_of_sound(self, c: float):
        pass

    @abstractmethod
    def add_roi(self, roi_data, generated: bool = False):
        pass

    @abstractmethod
    def rename_roi(self, old_name: Union[str, Tuple], new_name: str, new_position: str) -> None:
        """
        Rename a region of interest.

        Parameters
        ----------
        old_name : str or tuple
            Old roi name e.g. "tumour_left/0" or ("tumour_left", "0")
        new_name : str
            New roi name e.g. "brain"
        new_position : str
            New roi position e.g. "left"

        """
        pass

    @abstractmethod
    def delete_rois(self, name_position=None, number=None):
        pass

    @abstractmethod
    def delete_recons(self, name, recon_groups):
        pass

    @abstractmethod
    def delete_dso2s(self) -> None:
        pass
