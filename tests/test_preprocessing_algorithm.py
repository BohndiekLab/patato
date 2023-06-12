#  Copyright (c) Thomas Else 2023.
#  License: MIT

import unittest

import numpy as np

from patato import PreProcessor
from patato.core.image_structures.pa_time_data import PATimeSeries
from patato.data.get_example_datasets import get_msot_time_series_example

from patato.processing.gpu_preprocessing_algorithm import DefaultMSOTPreProcessor, GPUMSOTPreProcessor


class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.pa = get_msot_time_series_example("so2")[0:2, 0:2]

    def _test_preprocessor(self, pre_processor):
        detectors_start = self.pa.get_scan_geometry()

        time_factor = 3
        detector_factor = 2

        start_time_series = self.pa.get_time_series()

        preproc = pre_processor(time_factor=time_factor, detector_factor=detector_factor)

        new_t, d, _ = preproc.run(start_time_series, self.pa)

        self.assertIsInstance(d, dict)
        self.assertIsInstance(new_t, PATimeSeries)
        self.assertEqual(new_t.shape, start_time_series.shape[:-2] + (start_time_series.shape[-2] * detector_factor,
                                                                      start_time_series.shape[-1] * time_factor))
        self.assertTrue(np.all(np.isclose(detectors_start[0], d["geometry"][0])))
        return new_t

    def test_gpu_processing(self):
        try:
            import cupy as cp
        except ImportError:
            return None  # Skip test if cupy is not installed
        new_t = self._test_preprocessor(GPUMSOTPreProcessor)
        print(np.mean(new_t[0, 0].values))

    def test_overall_processing(self):
        new_t = self._test_preprocessor(DefaultMSOTPreProcessor)
        print(np.mean(new_t[0, 0].values))

    def test_jax_preprocessing(self):
        new_t = self._test_preprocessor(PreProcessor)
        print(np.mean(new_t[0, 0].values))
