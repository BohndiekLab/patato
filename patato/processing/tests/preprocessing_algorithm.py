#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import unittest
from os.path import split, join

import numpy as np
from ...core.image_structures.pa_time_data import PATimeSeries
from ...data.get_example_datasets import get_patato_data_folder
from ...io.msot_data import PAData

from ..gpu_preprocessing_algorithm import DefaultMSOTPreProcessor, GPUMSOTPreProcessor


class TestPreprocessing(unittest.TestCase):
    def test_gpu_processing(self):
        try:
            import cupy as cp
        except ImportError:
            return None  # Skip test if cupy is not installed
        data_folder = join(get_patato_data_folder(), "test")
        dummy_dataset = join(data_folder, "Scan_1.hdf5")

        pa = PAData.from_hdf5(dummy_dataset, "r+")[:, 0:2]

        detectors_start = pa.get_scan_geometry()

        time_factor = 3
        detector_factor = 2

        preproc = GPUMSOTPreProcessor(time_factor=time_factor, detector_factor=detector_factor)

        start_time_series = pa.get_time_series()

        new_t, d, _ = preproc.run(start_time_series, pa)

        self.assertIsInstance(d, dict)
        self.assertIsInstance(new_t, PATimeSeries)
        self.assertEqual(new_t.shape, start_time_series.shape[:-2] + (start_time_series.shape[-2] * detector_factor,
                                                                      start_time_series.shape[-1] * time_factor))
        self.assert_(np.all(np.isclose(detectors_start[0], d["geometry"][0])))

    def test_overall_processing(self):
        data_folder = join(get_patato_data_folder(), "test")
        dummy_dataset = join(data_folder, "Scan_1.hdf5")

        pa = PAData.from_hdf5(dummy_dataset, "r+")[:, 0:2]

        detectors_start = pa.get_scan_geometry()

        time_factor = 3
        detector_factor = 2

        preproc = DefaultMSOTPreProcessor(time_factor=time_factor, detector_factor=detector_factor)

        start_time_series = pa.get_time_series()

        new_t, d, _ = preproc.run(start_time_series, pa)

        self.assertIsInstance(d, dict)
        self.assertIsInstance(new_t, PATimeSeries)
        self.assertEqual(new_t.shape, start_time_series.shape[:-2] + (start_time_series.shape[-2] * detector_factor,
                                                                      start_time_series.shape[-1] * time_factor))
        self.assert_(np.all(np.isclose(detectors_start[0], d["geometry"][0])))
