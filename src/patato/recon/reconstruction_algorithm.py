#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import logging
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Optional, List

import numpy as np

from ..core.image_structures.pa_time_data import PATimeSeries
from ..core.image_structures.reconstruction_image import Reconstruction
from ..io.attribute_tags import ReconAttributeTags
from ..io.msot_data import PAData, HDF5Tags
from ..processing.processing_algorithm import TimeSeriesProcessingAlgorithm, ProcessingResult


class ReconstructionAlgorithm(TimeSeriesProcessingAlgorithm, ABC):
    def pre_prepare_data(self, x):
        return x

    def __init__(self,
                 n_pixels: Sequence[int],
                 field_of_view: Sequence[float],
                 **kwargs):
        super().__init__()
        self.n_pixels = n_pixels
        self.field_of_view = field_of_view
        self.custom_params = kwargs

    @abstractmethod
    def reconstruct(self, raw_data: np.ndarray,
                    fs: float,
                    geometry: np.ndarray,
                    n_pixels: Sequence[int],
                    field_of_view: Sequence[float],
                    speed_of_sound,
                    **kwargs) -> np.ndarray:
        pass

    @staticmethod
    def get_algorithm_name() -> str:
        pass

    def run(self, time_series: PATimeSeries,
            pa_data: PAData,
            speed_of_sound=None,
            geometry=None,
            **kwargs) -> Tuple[Reconstruction, dict, Optional[List[ProcessingResult]]]:
        from .. import PAT_MAXIMUM_BATCH_SIZE
        speed_of_sound = pa_data.get_speed_of_sound() if speed_of_sound is None else speed_of_sound

        if geometry is None:
            geometry = pa_data.get_scan_geometry()

        logging.debug(f"{time_series.attributes}, {geometry.shape}, {time_series.raw_data.shape}, {speed_of_sound}")

        irf = pa_data.get_impulse_response() if pa_data is not None else None
        wavelengths = time_series.da.coords.get("wavelengths") if pa_data is None else pa_data.get_wavelengths()

        # Process in batches to avoid GPU running out of memory.
        if time_series.shape[0] * time_series.shape[1] > PAT_MAXIMUM_BATCH_SIZE != -1:
            new_recons = []
            ts_raw = time_series.raw_data
            shape = ts_raw.shape
            ts_raw = ts_raw.reshape((-1,) + shape[-2:])
            for i in range(0, ts_raw.shape[0], PAT_MAXIMUM_BATCH_SIZE):
                raw = self.reconstruct(ts_raw[i:i + PAT_MAXIMUM_BATCH_SIZE], time_series.attributes["fs"],
                                       geometry, self.n_pixels, self.field_of_view, speed_of_sound,
                                       irf=irf, **kwargs, **self.custom_params)
                new_recons.append(np.asarray(raw))
            raw_data = np.concatenate(new_recons, axis=0).reshape(shape[:2] + new_recons[0].shape[1:])
        else:
            raw_data = self.reconstruct(time_series.raw_data,
                                        time_series.attributes["fs"],
                                        geometry,
                                        self.n_pixels,
                                        self.field_of_view,
                                        speed_of_sound,
                                        irf=irf,
                                        **kwargs, **self.custom_params)

        output_data = Reconstruction(raw_data, wavelengths,
                                     hdf5_sub_name=self.get_algorithm_name(),
                                     field_of_view=self.field_of_view)
        output_data.attributes[HDF5Tags.SPEED_OF_SOUND] = speed_of_sound
        output_data.attributes[ReconAttributeTags.RECONSTRUCTION_ALGORITHM] = self.get_algorithm_name()
        output_data.attributes[ReconAttributeTags.X_NUMBER_OF_PIXELS] = self.n_pixels[0]
        output_data.attributes[ReconAttributeTags.Y_NUMBER_OF_PIXELS] = self.n_pixels[1]
        output_data.attributes[ReconAttributeTags.Z_NUMBER_OF_PIXELS] = self.n_pixels[2]
        output_data.attributes[ReconAttributeTags.X_FIELD_OF_VIEW] = self.field_of_view[0]
        output_data.attributes[ReconAttributeTags.Y_FIELD_OF_VIEW] = self.field_of_view[1]
        output_data.attributes[ReconAttributeTags.Z_FIELD_OF_VIEW] = self.field_of_view[2]
        output_data.attributes[ReconAttributeTags.ADDITIONAL_PARAMETERS] = kwargs
        output_data.attributes[HDF5Tags.WAVELENGTH] = wavelengths

        for a in time_series.attributes:
            output_data.attributes[a] = time_series.attributes[a]

        return output_data, {}, None
