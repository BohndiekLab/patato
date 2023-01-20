#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import h5py
import numpy as np
from ...core.image_structures.image_structure_types import IMAGE_DATA_TYPES
from ...core.image_structures.pa_time_data import PATimeSeries
from ...io.attribute_tags import HDF5Tags, ROITags, UnmixingAttributeTags, ReconAttributeTags
from ...io.hdf.fileimporter import WriterInterface, ReaderInterface
from ...utils.rois.roi_type import ROI
import json

if TYPE_CHECKING:
    import numpy as np
import dask.array as da


def renumber_group(dataset: h5py.Group) -> None:
    """
    Renumber an old roi group.

    Parameters
    ----------
    dataset : h5py.Group
    """
    # Clean up old group:
    original_names = sorted(list(dataset.keys()), key=int)
    new_names = [str(x) for x in range(len(original_names))]
    for old, new in zip(original_names, new_names):
        dataset.move(old, new)


# Might be useful to set enviro variable: HDF5_USE_FILE_LOCKING=FALSE
def load_image_from_hdf5(cls, dataset, file):
    """

    Parameters
    ----------
    cls
    dataset
    file

    Returns
    -------

    """
    ax_1_meaning = cls.get_ax1_label_meaning()
    dataset_da = da.from_array(dataset, chunks=(1,) + dataset.shape[1:])

    if ax_1_meaning == HDF5Tags.WAVELENGTH:
        ax_1_labels = dataset.attrs.get(HDF5Tags.WAVELENGTH, file["wavelengths"])
    elif ax_1_meaning == "SPECTRA":
        ax_1_labels = dataset.attrs["SPECTRA"]
    elif ax_1_meaning is None:
        ax_1_labels = None
    elif dataset_da.shape[1] == 1:
        ax_1_labels = np.array([dataset.name.split("/")[-3]])
    else:
        ax_1_labels = np.arange(dataset_da.shape[1])

    fov = [dataset.attrs.get(fx, None) for fx in [ReconAttributeTags.X_FIELD_OF_VIEW,
                                                  ReconAttributeTags.Y_FIELD_OF_VIEW,
                                                  ReconAttributeTags.Z_FIELD_OF_VIEW]]

    if all([x is None for x in fov]):
        fov_x = dataset.attrs.get(ReconAttributeTags.OLD_FIELD_OF_VIEW, 1.)
        fov = [fov_x, fov_x, fov_x]

    new_cls = cls(dataset_da, ax_1_labels,
                  algorithm_id=dataset.attrs.get(UnmixingAttributeTags.SUFFIX, ""),
                  attributes=dict(dataset.attrs),
                  field_of_view=fov)
    if not cls.is_single_instance():
        new_cls.hdf5_sub_name = dataset.name.split("/")[-2]
    return new_cls


class HDF5Writer(WriterInterface):
    def delete_images(self, image):
        pass

    def __init__(self, file):
        WriterInterface.__init__(self)
        if type(file) == h5py.File:
            self.file = file
        else:
            self.file = h5py.File(file, "a")
        self.overwrite = False  # TODO: implement this.

    def set_scan_datetime(self, datetime):
        self.file.attrs[HDF5Tags.DATE] = str(datetime)

    def set_pa_data(self, raw_data):
        if type(raw_data) == PATimeSeries:
            raw_data = raw_data.da
        self.file.create_dataset(HDF5Tags.RAW_DATA, data=raw_data, dtype=np.uint16)

    def set_scan_name(self, scan_name: str):
        self.file.attrs[HDF5Tags.SCAN_NAME] = scan_name
        self.file[HDF5Tags.RAW_DATA].attrs[HDF5Tags.SCAN_NAME] = scan_name

    def set_temperature(self, temperature: "np.ndarray"):
        self.file.create_dataset(HDF5Tags.TEMPERATURE, data=temperature, compression="gzip")

    def set_correction_factor(self, correction_factor):
        self.file.create_dataset(HDF5Tags.OVERALL_CORR, data=correction_factor, compression="gzip")

    def set_scanner_z_position(self, z_position):
        self.file.create_dataset(HDF5Tags.Z_POSITION, data=z_position, compression="gzip")

    def set_run_numbers(self, run_numbers):
        self.file.create_dataset(HDF5Tags.RUN, data=run_numbers, compression="gzip")

    def set_repetition_numbers(self, repetition_numbers):
        self.file.create_dataset(HDF5Tags.REPETITION, data=repetition_numbers, compression="gzip")

    def set_scan_times(self, scan_times):
        self.file.create_dataset(HDF5Tags.TIMESTAMP, data=scan_times, compression="gzip")

    def set_sensor_geometry(self, sensor_geometry):
        self.file.create_dataset(HDF5Tags.SCAN_GEOMETRY, data=sensor_geometry, compression="gzip")

    def set_us_data(self, us_data, us_fov):
        # At some point convert this to image sequence data types.
        self.file.create_dataset(HDF5Tags.ULTRASOUND, data=us_data)
        self.file[HDF5Tags.ULTRASOUND].attrs[HDF5Tags.ULTRASOUND_FIELD_OF_VIEW] = us_fov

    def set_impulse_response(self, impulse_response):
        self.file.create_dataset(HDF5Tags.IMPULSE_RESPONSE, data=impulse_response, compression="gzip")

    def set_wavelengths(self, wavelengths):
        self.file.create_dataset(HDF5Tags.WAVELENGTH, data=wavelengths)

    def set_us_offsets(self, us_offsets):
        if us_offsets is not None:
            self.file.create_dataset(HDF5Tags.ULTRASOUND_FRAME_OFFSET, data=us_offsets,
                                     compression="gzip")

    def set_water_absorption(self, water_absorption, pathlength):
        # TODO: Maybe refactor this?
        water = self.file.create_dataset(HDF5Tags.WATER_ABSORPTION_COEFF, data=water_absorption,
                                         compression="gzip")
        water.attrs[HDF5Tags.WATER_PATHLENGTH] = pathlength

    def add_image(self, image_data):
        # Image data can be reconstructions, unmixed data etc.
        grp = self.file.require_group(image_data.get_hdf5_group_name())  # E.g. unmixed or recons
        subgroup = grp.require_group(image_data.get_hdf5_sub_name())  # E.g. reconstruction method or something
        name = str(len(subgroup.keys())) + image_data.algorithm_id  # E.g. "0" + "E"
        dataset = subgroup.create_dataset(name, data=image_data.raw_data)

        # Copy attributes
        for a in image_data.attributes:
            if type(image_data.attributes[a]) == dict:
                for b in image_data.attributes[a]:
                    attr = image_data.attributes[a][b]
                    if type(attr) == str:
                        attr = bytes(attr, "utf-8")
                    try:
                        dataset.attrs[b] = attr
                    except TypeError:
                        dataset.attrs[b] = json.dumps(attr)
            else:
                attr = image_data.attributes[a]
                if type(attr) == str:
                    attr = bytes(attr, "utf-8")
                from numpy import nan, ndarray
                if attr is None:
                    attr = nan
                if type(attr) is ndarray:
                    if attr.dtype == "<U4":
                        attr = [bytes(a, "utf-8") for a in attr]
                dataset.attrs[a] = attr

    def set_scan_comment(self, comment: str):
        self.file.attrs[HDF5Tags.SCAN_COMMENT] = comment

    def set_sampling_frequency(self, frequency: float):
        self.file[HDF5Tags.RAW_DATA].attrs[HDF5Tags.SAMPLING_FREQ] = frequency
        self.file.attrs[HDF5Tags.SAMPLING_FREQ] = frequency

    def set_segmentation(self, seg):
        self.file.create_dataset(HDF5Tags.SEGMENTATION, data=seg, compression="gzip")

    def set_speed_of_sound(self, c: float):
        self.file[HDF5Tags.RAW_DATA].attrs[HDF5Tags.SPEED_OF_SOUND] = c
        self.file.attrs[HDF5Tags.SPEED_OF_SOUND] = c

    def add_roi(self, roi_data: ROI, generated: bool = False):
        """
        Add a region of interest to the hdf5 file.

        Parameters
        ----------
        roi_data : ROI
        generated : bool, default False
        """
        roi_group = self.file.require_group(HDF5Tags.REGIONS_OF_INTEREST)
        print(roi_data.attributes[ROITags.ROI_NAME] + "_" +
              roi_data.attributes[ROITags.ROI_POSITION])
        region_group = roi_group.require_group(roi_data.attributes[ROITags.ROI_NAME] + "_" +
                                               roi_data.attributes[ROITags.ROI_POSITION])
        n = str(len(region_group))
        if type(roi_data.points) in [np.ndarray, list]:
            dataset = region_group.create_dataset(n, data=roi_data.points)
        else:
            raise NotImplementedError("Cannot save specialist roi type (yet...).")
        # Add attributes
        for attr in roi_data.attributes:
            dataset.attrs[attr] = roi_data.attributes[attr]
        dataset.attrs[ROITags.GENERATED_ROI] = generated

    def rename_roi(self, old_name, new_name, new_position):
        if type(old_name) == str:
            old_name = tuple(old_name.split("/"))
        roi_dataset = self.file[HDF5Tags.REGIONS_OF_INTEREST]
        roi_dataset["/".join(old_name)].attrs[ROITags.ROI_NAME] = new_name
        roi_dataset["/".join(old_name)].attrs[ROITags.ROI_POSITION] = new_position

        new_dataset_name = new_name + "_" + new_position
        new_number = "0"

        if new_dataset_name in roi_dataset:
            new_number = str(len(roi_dataset[new_dataset_name]))
        roi_dataset.move("/".join(old_name), new_dataset_name + "/" + new_number)

        renumber_group(roi_dataset[old_name[0]])

    def delete_rois(self, name_position=None, number=None):
        if "/" in name_position and number is None:
            name, number = name_position.split("/")
            name_position = name

        if HDF5Tags.REGIONS_OF_INTEREST in self.file:
            if name_position is None and number is None:
                del self.file[HDF5Tags.REGIONS_OF_INTEREST]
            elif number is None and name_position in self.file[HDF5Tags.REGIONS_OF_INTEREST]:
                del self.file[HDF5Tags.REGIONS_OF_INTEREST][name_position]
            elif name_position in self.file[HDF5Tags.REGIONS_OF_INTEREST]:
                if number in self.file[HDF5Tags.REGIONS_OF_INTEREST][name_position]:
                    del self.file[HDF5Tags.REGIONS_OF_INTEREST][name_position][number]
                    # Renumber rois.
                    renumber_group(self.file[HDF5Tags.REGIONS_OF_INTEREST][name_position])

    def delete_recons(self, name=None, recon_groups: Optional[Sequence[str]] = None):
        """
        Delete some reconstructions.

        Parameters
        ----------
        name : str or None
        recon_groups : (iterable of str) or None
        """
        if recon_groups is None:
            recon_groups = [HDF5Tags.RECONSTRUCTION,
                            HDF5Tags.UNMIXED,
                            HDF5Tags.SO2,
                            HDF5Tags.THB]
        if name is None:
            for group in recon_groups:
                if group in self.file:
                    del self.file[group]
        else:
            for group in recon_groups:
                if group in self.file:
                    for recon_name in self.file[group]:
                        if name == recon_name:
                            del self.file[group][name]

    def delete_dso2s(self):
        """
        Delete the old dso2 datasets.
        Might want to remove this or refactor it in some way?
        """
        for group in [HDF5Tags.DELTA_SO2, HDF5Tags.BASELINE_SO2_STANDARD_DEVIATION, HDF5Tags.BASELINE_SO2,
                      HDF5Tags.DELTA_ICG, HDF5Tags.BASELINE_ICG, HDF5Tags.BASELINE_ICG_SIGMA]:
            if group in self.file:
                del self.file[group]


class HDF5Reader(ReaderInterface):
    def _get_rois(self):
        output = {}
        if HDF5Tags.REGIONS_OF_INTEREST in self.file:
            for roi_name in self.file[HDF5Tags.REGIONS_OF_INTEREST]:
                roi_group = self.file[HDF5Tags.REGIONS_OF_INTEREST][roi_name]
                for roi_number in roi_group:
                    dataset = roi_group[roi_number]
                    clinical = self.is_clinical()
                    frame_type = ROITags.Z_POSITION if not clinical else ROITags.REPETITION
                    match_frames = self.get_scanner_z_position() if not clinical else self.get_repetition_numbers()
                    ax0_indices = np.where(np.isclose(match_frames[:, 0], dataset.attrs.get(frame_type, 1.0)))[0]
                    output[(roi_name, roi_number)] = ROI(dataset[:], dataset.attrs[ROITags.Z_POSITION],
                                                         dataset.attrs[ROITags.RUN],
                                                         dataset.attrs.get(ROITags.REPETITION, np.nan),
                                                         dataset.attrs[ROITags.ROI_NAME],
                                                         dataset.attrs[ROITags.ROI_POSITION],
                                                         dataset.attrs.get(ROITags.GENERATED_ROI, False),
                                                         ax0_indices)
        return output

    def __init__(self, file):
        ReaderInterface.__init__(self)
        if type(file) == h5py.File:
            self.file = file
        else:
            self.file = h5py.File(file, "r")

    def get_scan_datetime(self):
        import dateutil.parser
        try:
            if type(self.file.attrs[HDF5Tags.DATE]) == str:
                return dateutil.parser.isoparse(self.file.attrs[HDF5Tags.DATE]).replace(tzinfo=None)
            else:
                return self.file.attrs[HDF5Tags.DATE]
        except KeyError:
            return np.nan

    def _get_pa_data(self):
        return self.file[HDF5Tags.RAW_DATA], dict(self.file[HDF5Tags.RAW_DATA].attrs)

    def get_scan_name(self):
        return self.file[HDF5Tags.RAW_DATA].attrs[HDF5Tags.SCAN_NAME]

    def _get_temperature(self):
        return self.file[HDF5Tags.TEMPERATURE]

    def _get_correction_factor(self):
        if np.any(np.isnan(self.file[HDF5Tags.OVERALL_CORR][:])):
            # Old version
            return self.file["POWER"][:]
        return self.file[HDF5Tags.OVERALL_CORR][:]

    def _get_scanner_z_position(self):
        return self.file[HDF5Tags.Z_POSITION]

    def _get_run_numbers(self):
        return self.file[HDF5Tags.RUN]

    def _get_repetition_numbers(self):
        return self.file[HDF5Tags.REPETITION]

    def _get_scan_times(self):
        return self.file[HDF5Tags.TIMESTAMP][:, :] * 1e-7

    def _get_sensor_geometry(self):
        return self.file[HDF5Tags.SCAN_GEOMETRY][:]

    def get_us_data(self):
        if HDF5Tags.ULTRASOUND in self.file:
            return self.file.get(HDF5Tags.ULTRASOUND, None), \
                   self.file[HDF5Tags.ULTRASOUND].attrs[HDF5Tags.ULTRASOUND_FIELD_OF_VIEW]
        else:
            return None, {}

    def get_us_offsets(self):
        return self.file[HDF5Tags.ULTRASOUND_FRAME_OFFSET]

    def get_impulse_response(self):
        return self.file[HDF5Tags.IMPULSE_RESPONSE]

    def _get_wavelengths(self):
        return self.file[HDF5Tags.WAVELENGTH][:]

    def _get_water_absorption(self):
        return (self.file[HDF5Tags.WATER_ABSORPTION_COEFF],
                self.file[HDF5Tags.WATER_ABSORPTION_COEFF].attrs[HDF5Tags.WATER_PATHLENGTH])

    def _get_datasets(self):
        output = {}
        for image_type in IMAGE_DATA_TYPES:
            output[image_type] = {}
            dtype = IMAGE_DATA_TYPES[image_type]
            images = []
            if image_type in self.file:
                # Loop through all types of this image and add them to the list
                for recon in self.file[image_type]:
                    for recon_num in self.file[image_type][recon]:
                        image = load_image_from_hdf5(dtype, self.file[image_type][recon][recon_num], self.file)
                        images.append(((recon, recon_num), image))
            for (recon, num), data in images:
                output[image_type][(recon, num)] = data
        return output

    def get_scan_comment(self):
        return self.file.attrs[HDF5Tags.SCAN_COMMENT]

    def _get_sampling_frequency(self):
        return self.file[HDF5Tags.RAW_DATA].attrs[HDF5Tags.SAMPLING_FREQ]

    def get_speed_of_sound(self):
        return self.file[HDF5Tags.RAW_DATA].attrs.get(HDF5Tags.SPEED_OF_SOUND, None)

    def _get_segmentation(self):
        return self.file.get(HDF5Tags.SEGMENTATION, None)


class PACFISHInterface(ReaderInterface):
    pass
