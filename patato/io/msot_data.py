#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from __future__ import annotations

import copy
from typing import Union, Dict, Tuple, Optional, Sequence, TYPE_CHECKING, Type

import h5py
import numpy as np
import xarray

from .hdf.fileimporter import ReaderInterface, WriterInterface
from .hdf.hdf5_interface import HDF5Reader, HDF5Writer
from ..core.image_structures.pa_fourier_data import PAFourierDomain
from ..core.image_structures.pa_time_data import PATimeSeries
from ..core.image_structures.single_image import SingleImage
from ..core.image_structures.single_parameter_data import SingleParameterData
from ..utils.roi_operations import get_rim_core_rois
from ..utils.rois.roi_type import ROI

if TYPE_CHECKING:
    try:
        from pyopencl.array import Array
    except ImportError:
        Array = np.ndarray
    from ..core.image_structures.image_sequence import ImageSequence

from functools import lru_cache

from ..io.attribute_tags import HDF5Tags


class PAData:
    """A class that contains the interface to access data from a single scan. Any source of scans (e.g.
    iThera/HDF5/IPASC) can be linked to this.
    """
    @property
    def shape(self) -> Tuple[int]:
        """
        Returns the shape of the dataset, minus the image size.

        Returns
        -------
        tuple of int
            Shape of the dataset.
        """
        return self.get_time_series().shape[:-2]

    def __getitem__(self, item: Union[slice, Tuple, None]) -> "PAData":
        """
        Slice the photoacoustic data. Choose a particular frame/wavelength etc.

        Parameters
        ----------
        item : slice or tuple

        Returns
        -------
        PAData
            Sliced pa data.
        """
        new_data = self.copy()
        new_data.scan_reader = copy.copy(new_data.scan_reader)[item]
        new_data.scan_writer = new_data.scan_writer
        return new_data

    def __init__(self, scan_reader: ReaderInterface, scan_writer: WriterInterface = None) -> None:
        """

        Parameters
        ----------
        scan_reader
        scan_writer
        """
        super().__init__()
        self.scan_reader = scan_reader
        self.scan_writer = scan_writer

        self.default_recon = None
        self.default_unmixing_type = ""
        self.external_roi_interface = None

    def is_clinical(self):
        return self.scan_reader.is_clinical()

    def set_default_recon(self, rec_name: Optional[str] = None) -> None:
        """
        Make all returned data be of a particular reconstruction type.
        It is recommended to run this at the start of analysis scripts.

        Parameters
        ----------
        rec_name : tuple of str, optional
        """
        if rec_name is None and self.default_recon is None:
            rec_name = list(self.get_scan_reconstructions().keys())[0]
        if not (self.default_recon is not None and rec_name is None):
            self.default_recon = rec_name

    def copy(self, cls: Optional[Type["PAData"]] = None) -> PAData:
        """
        Copy the pa data with changes given.

        Parameters
        ----------
        cls : Type[PAData]

        Returns
        -------
        PAData
            Copy of the dataset.
        """
        if cls is None:
            cls = self.__class__

        from copy import copy
        c = copy(self)
        c.__class__ = cls
        return c

    def get_scan_name(self) -> str:
        """
        Get the scan name.

        Returns
        -------
        str
            Scan name.

        """
        return self.scan_reader.get_scan_name()

    def get_scan_datetime(self):
        return self.scan_reader.get_scan_datetime()

    def get_sampling_frequency(self) -> float:
        """
        Get the scan's sampling frequency.

        Returns
        -------
        float
            Sampling Frequency
        """
        return self.scan_reader.get_sampling_frequency()

    def get_overall_correction_factor(self) -> Union[np.ndarray, Array, xarray.DataArray]:
        """
        Return the energy correction factors for the dataset.

        Returns
        -------
        np.ndarray
            Overall correction factor.
        """
        return self.scan_reader.get_correction_factor()

    def get_impulse_response(self) -> Union[np.ndarray, Array]:
        """
        Return the time-domain impulse response function.

        Returns
        -------
        np.ndarray or pyopencl.array.Array
            Impulse response function.
        """
        return self.scan_reader.get_impulse_response()

    def get_n_samples(self) -> int:
        """
        Get the number of time samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.get_time_series().shape[-1]

    def get_speed_of_sound(self) -> Union[float, None]:
        """
        Get the speed of sound of the data if it has been set.

        Returns
        -------
        float or None
            Speed of sound
        """
        return self.scan_reader.get_speed_of_sound()

    def get_scan_geometry(self) -> Union[np.ndarray, Array]:
        """
        Get the scan detector geometry.

        Returns
        -------
        np.ndarray or pyopencl.array.Array
        """
        return self.scan_reader.get_sensor_geometry()

    def get_wavelengths(self) -> np.ndarray:
        """
        Get the wavelengths used in the scan.

        Returns
        -------
        np.ndarray
            Scan Wavelengths.
        """
        return self.scan_reader.get_wavelengths()

    def get_scan_images(self, group: str, ignore_default=False, suffix="") -> Union[
        Dict[Tuple[str, str], ImageSequence], ImageSequence]:
        """
        Get the scan images, e.g. reconstructions or so2 etc.

        Parameters
        ----------
        group : str
            Group to get images from.
        ignore_default : bool
            Ignore the default reconstruction.
        suffix : str
            Suffix to add to the image number (e.g. for ICG unmixing).

        Returns
        -------
        (dict of {tuple of (str, str): ImageSequence}) or ImageSequence
            Images of certain type if default recon has been set, or dict or images for all reconstructions.
        """

        datasets = self.scan_reader.get_datasets()
        if group not in datasets:
            return {}
        if self.default_recon is None or ignore_default:
            return datasets.get(group, {})
        else:
            image = (self.default_recon[0], self.default_recon[1] + suffix)
            return datasets.get(group, {}).get(image, {})

    def get_scan_reconstructions(self):
        return self.get_scan_images(HDF5Tags.RECONSTRUCTION)

    def get_scan_unmixed(self):
        return self.get_scan_images(HDF5Tags.UNMIXED, suffix=self.default_unmixing_type)

    def get_scan_so2(self):
        return self.get_scan_images(HDF5Tags.SO2)

    def close(self):
        self.scan_reader.close()
        if self.scan_writer is not None:
            self.scan_writer.close()

    def get_scan_mean(self, dataset: ImageSequence, operation=np.mean):
        """

        Parameters
        ----------
        dataset
        operation

        Returns
        -------

        """
        if type(dataset) == dict:
            raise NotImplementedError
        new_dataset = SingleImage(operation(dataset.raw_data, axis=0)[0], dataset.ax_1_labels,
                                  field_of_view=dataset.fov_3d, attributes=dataset.attributes)
        return new_dataset

    def get_scan_so2_time_mean(self):
        """

        Returns
        -------

        """
        return self.get_scan_mean(self.get_scan_so2())

    def get_scan_thb_time_mean(self):
        """

        Returns
        -------

        """
        return self.get_scan_mean(self.get_scan_thb())

    def get_scan_so2_time_standard_deviation(self):
        """

        Returns
        -------

        """
        return self.get_scan_mean(self.get_scan_so2(), np.std)

    def get_scan_thb(self):
        """

        Returns
        -------

        """
        return self.get_scan_images(HDF5Tags.THB)

    def get_scan_dso2(self):
        """

        Returns
        -------

        """
        return self.get_scan_images(HDF5Tags.DELTA_SO2)

    def get_scan_dicg(self):
        """

        Returns
        -------

        """
        return self.get_scan_images(HDF5Tags.DELTA_ICG)

    def get_scan_baseline_icg(self):
        """

        Returns
        -------

        """
        return self.get_scan_images(HDF5Tags.BASELINE_ICG)

    def get_scan_baseline_standard_deviation_icg(self):
        """

        Returns
        -------

        """
        return self.get_scan_images(HDF5Tags.BASELINE_ICG_SIGMA)

    def get_responding_pixels(self, nsigma=2):
        """

        Parameters
        ----------
        nsigma

        Returns
        -------

        """
        responding = None
        delta_images = None
        if self.get_scan_dicg():
            delta_images = self.get_scan_dicg()
            sigma_icg = self.get_scan_baseline_standard_deviation_icg()
            responding = delta_images.raw_data > nsigma * sigma_icg.raw_data
        elif self.get_scan_dso2():
            delta_images = self.get_scan_dso2()
            sigma_so2 = self.get_scan_baseline_standard_deviation_so2()
            responding = delta_images.raw_data > nsigma * sigma_so2.raw_data
        else:
            return None
        return SingleImage(responding, None, algorithm_id=delta_images.algorithm_id,
                           attributes=delta_images.attributes,
                           hdf5_sub_name=delta_images.hdf5_sub_name, field_of_view=delta_images.fov_3d)

    def get_scan_baseline_so2(self):
        """

        Returns
        -------

        """
        return self.get_scan_images(HDF5Tags.BASELINE_SO2)

    def get_scan_baseline_standard_deviation_so2(self):
        """

        Returns
        -------

        """
        return self.get_scan_images(HDF5Tags.BASELINE_SO2_STANDARD_DEVIATION)

    # Some cycling hypoxia analysis here.
    @lru_cache
    def get_scan_so2_frequency_components(self, do_detrend=True, fmin=1e-5, fmax=1000, fnum=1000):
        """

        Parameters
        ----------
        do_detrend
        fmin
        fmax
        fnum

        Returns
        -------

        """
        from scipy.signal import detrend, lombscargle
        so2 = self.get_scan_images(HDF5Tags.SO2)
        if type(so2) == dict:
            if len(so2) == 1:
                so2 = so2[list(so2.keys())[0]]
            else:
                raise NotImplementedError("""Frequency components are only enabled when there is one
                reconstruction set. Run PAData.set_default_recon() before running this if in doubt.
                                          """)

        detrended = detrend(so2.raw_data, axis=0, type="linear" if do_detrend else "constant")

        times = self.get_timestamps()[:, 0].copy()
        times -= times[0]

        frequencies = np.linspace(fmin, fmax, fnum)

        def lomb(a, b, c):
            return lombscargle(a, b.copy(), c)

        so2_frequency = SingleParameterData(np.apply_along_axis(lambda x: lomb(times, x, frequencies * 2 * np.pi),
                                                                0, detrended),
                                            so2.ax_1_labels,
                                            field_of_view=so2.fov_3d, attributes=so2.attributes)
        so2_frequency.da.coords["frames"] = frequencies * 2 * np.pi
        return so2_frequency

    def get_scan_so2_frequency_peak(self, fnum=1000):
        """

        Parameters
        ----------
        fnum

        Returns
        -------

        """
        so2 = self.get_scan_so2_frequency_components(fnum=fnum)
        raw_data = np.max(so2.raw_data, axis=0)[0]
        so2_frequency = SingleImage(raw_data,
                                            so2.ax_1_labels,
                                            field_of_view=so2.fov_3d, attributes=so2.attributes)
        return so2_frequency

    def get_scan_so2_frequency_sum(self, fnum=1000):
        """

        Parameters
        ----------
        fnum

        Returns
        -------

        """
        so2 = self.get_scan_so2_frequency_components(fnum=fnum)
        raw_data = np.sum(so2.raw_data, axis=0)[0]
        so2_frequency = SingleImage(raw_data,
                                            so2.ax_1_labels,
                                            field_of_view=so2.fov_3d, attributes=so2.attributes)
        return so2_frequency

    def get_segmentation(self):
        """

        Returns
        -------

        """
        return self.scan_reader.get_segmentation()

    def get_time_series(self) -> PATimeSeries:
        """

        Returns
        -------

        """
        dataset = self.scan_reader.get_pa_data()
        if type(dataset) is PAFourierDomain:
            return dataset.to_time_domain()
        elif type(dataset) is not PATimeSeries:
            raise ValueError("raw_data attribute must be either TimeSeries or FourierDomain type.")
        else:
            return dataset

    def get_fft(self) -> PAFourierDomain:
        """

        Returns
        -------

        """
        dataset = self.scan_reader.get_pa_data()
        if type(dataset) is PATimeSeries:
            return dataset.to_fourier_domain()
        elif not type(dataset) is PAFourierDomain:
            raise ValueError("raw_data attribute must be either TimeSeries or FourierDomain type.")
        else:
            return dataset

    def get_recon_types(self) -> list:
        """
        Get the list of different reconstruction types.

        Returns
        -------
        list
            List of different reconstruction types that we have.
        """

        return list(self.get_scan_images(HDF5Tags.RECONSTRUCTION, True).keys())

    def get_z_positions(self) -> np.ndarray:
        """
        Get the z-positions of the sensor.

        Returns
        -------
        np.ndarray
            Z-positions array.
        """
        return self.scan_reader.get_scanner_z_position()

    def get_run_number(self) -> np.ndarray:
        """
        Get the run number for each of the frames.

        Returns
        -------
        np.ndarray
            Get the run numbers of each of the frames.

        """
        return self.scan_reader.get_run_numbers()

    def get_repetition_numbers(self) -> np.ndarray:
        """
        Get the scan repetition numbers for each frame.

        Returns
        -------
        np.ndarray
            Scan repetition numbers.
        """
        return self.scan_reader.get_repetition_numbers()

    def get_timestamps(self) -> np.ndarray:
        """
        Get the scan timestamps in seconds.

        Returns
        -------
        np.ndarray
            Timestamps in seconds.
        """
        # in seconds
        return self.scan_reader.get_scan_times()

    def get_rois(self, filter_rois=None,
                 interpolate: bool = False,
                 get_rim_cores=None,
                 rim_core_distance=None) -> Dict[Tuple[str, str], "ROI"]:
        """
        Get the regions of interest from the dataset.

        Parameters
        ----------
        rim_core_distance
        get_rim_cores
        filter_rois: dict or None
        interpolate: bool

        Returns
        -------
        dict of {(tuple of (str, str): ROI}
            Return all the rois.

        """
        if self.external_roi_interface is not None:
            reader = self.external_roi_interface.scan_reader
        else:
            reader = self.scan_reader
        output_rois = reader.get_rois(interpolate)

        if get_rim_cores is not None:
            rim_core_rois = {}
            for o in output_rois:
                if o[0] in get_rim_cores:
                    core, rim = get_rim_core_rois(output_rois[o], rim_core_distance)
                    name, position = o[0].split("_")
                    rim_core_rois[(name + ".core_" + position, o[1])] = core
                    rim_core_rois[(name + ".rim_" + position, o[1])] = rim
            output_rois.update(rim_core_rois)
        if filter_rois is not None:
            new_out_rois = {}
            for k in output_rois:
                for f in filter_rois:
                    if f == k[0] or f + "_" == k[0]:
                        new_out_rois[k] = output_rois[k]
            output_rois = new_out_rois
        return output_rois

    def delete_recons(self, name=None, recon_groups: Optional[Sequence[str]] = None):
        """
        Delete the reconstructions.

        Parameters
        ----------
        name : str or None
        recon_groups : (iterable of str) or None
        """
        if self.scan_writer is None:
            raise NotImplementedError("Deletion only possible with a writing interface.")
        self.scan_writer.delete_recons(name, recon_groups)

    def set_speed_of_sound(self, c: float) -> None:
        """
        Change the speed of sound for the dataset.

        Parameters
        ----------
        c : float
            Speed of sound.

        """
        if self.scan_writer is None:
            raise NotImplementedError("No writing capability enabled.")
        self.scan_writer.set_speed_of_sound(c)

    def delete_rois(self, name_position: Optional[str] = None, number: Optional[str] = None) -> None:
        """
        Delete a roi with name and number. If number is None, will delete all.
        If name and number is None, delete all.

        Parameters
        ----------
        name_position
            str or None
        number
            str or none
        """
        self.scan_writer.delete_rois(name_position, number)

    def add_roi(self, roi_data: "ROI", generated: bool = False) -> None:
        """
        Add a region of interest to the hdf5 file.

        Parameters
        ----------
        roi_data : ROI
        generated : bool, default False
        """
        self.scan_writer.add_roi(roi_data, generated)

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
        self.scan_writer.rename_roi(old_name, new_name, new_position)

    def clear_dso2(self) -> None:
        self.scan_writer.delete_dso2s()

    @classmethod
    def from_hdf5(cls, filename: Union[str, h5py.File], mode: str = "r"):
        if type(filename) == str:
            file = h5py.File(filename, mode)
        else:
            file = filename
        return cls(HDF5Reader(file), HDF5Writer(file))

    def save_hdf5(self, filename: str, update=False):
        file = h5py.File(filename, "a")
        writer = HDF5Writer(file)
        return writer.save_file(self.scan_reader, update=update)

    def save_to_hdf5(self, filename, update=False):
        return self.save_hdf5(filename, update)

    @property
    def dataset(self):
        """

        Returns
        -------

        """
        return self.get_time_series()

    def summary_measurements(self, metrics=None,
                             include_rois=None, roi_kwargs=None, just_summary=True, return_masks=False):
        """

        Parameters
        ----------
        metrics
        include_rois
        roi_kwargs
        just_summary

        Returns
        -------

        """
        import pandas as pd
        if metrics is None:
            metrics = ["thb", "so2"]
        if roi_kwargs is None:
            roi_kwargs = {}

        rois = self.get_rois(**roi_kwargs)

        if include_rois is not None:
            new_rois = {}
            for x in rois:
                if x[0][:-1] in include_rois or x[0] in include_rois:
                    new_rois[x] = rois[x]
            rois = new_rois

        measurements = []
        for m in metrics:
            if m == "thb":
                measurements.append(self.get_scan_thb())
            elif m == "so2":
                measurements.append(self.get_scan_so2())
            elif m == "icg":
                unmixed = self.get_scan_unmixed()
                n = [i for i, l in enumerate(unmixed.ax_1_labels) if l.lower() == "icg"][0]
                measurements.append(unmixed[:, n: n + 1])
            elif m == "dso2":
                measurements.append(self.get_scan_dso2())
            elif m == "baseline_so2":
                measurements.append(self.get_scan_baseline_so2())
            elif m == "responding":
                measurements.append(self.get_responding_pixels())
            elif m == "recons":
                measurements.append(self.get_scan_reconstructions())
            else:
                raise NotImplementedError(f"Metric {m} not yet implemented.")

        if not rois:
            return pd.DataFrame({})
        outputs = []
        for name, roi in rois.items():
            output_roi = {}
            for i, m in enumerate(measurements):
                if m:
                    mask, data_slice = roi.to_mask_slice(m)
                    output_roi[metrics[i]] = data_slice.raw_data.T[mask.T].T
                else:
                    print(f"Skipping metric {metrics[i]}")

            mask, _, selection = roi.to_mask_slice(self.get_scan_reconstructions(), return_selection=True)
            if return_masks:
                return_mask, _ = roi.to_mask_slice(self.get_scan_so2())
                output_roi["Mask"] = return_mask
            output_roi["Timings"] = self.get_timestamps()[selection]
            output_roi["Area"] = np.sum(mask)
            for a in roi.attributes:
                output_roi[a] = roi.attributes[a]
            output_roi["number"] = name[1]
            output_roi["name"] = name
            output_roi["Wavelengths"] = self.get_wavelengths()
            outputs.append(output_roi)

        output_table = pd.DataFrame(outputs)

        summary_methods = {"mean": np.mean,
                           "median": np.median,
                           "std": np.std}
        for name, method in summary_methods.items():
            for metric in metrics:
                if metric in output_table.columns:
                    output_table[metric + "_" + name] = output_table[metric].apply(
                        lambda t: np.squeeze(method(t, axis=-1))[()])

        if just_summary:
            for metric in metrics:
                if metric in output_table.columns:
                    del output_table[metric]

        return output_table
