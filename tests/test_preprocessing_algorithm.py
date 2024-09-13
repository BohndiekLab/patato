#  Copyright (c) Thomas Else 2023.
#  License: MIT

import unittest
from importlib.util import find_spec

import numpy as np

from patato import PreProcessor, PAData
from patato.core.image_structures.pa_time_data import PATimeSeries

from patato.processing.gpu_preprocessing_algorithm import GPUMSOTPreProcessor
from patato.processing.preprocessing_algorithm import NumpyPreProcessor


class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.pa = PAData.from_hdf5("test_data.hdf5")

    def _test_preprocessor(self, pre_processor):
        detectors_start = self.pa.get_scan_geometry()

        time_factor = 3
        detector_factor = 2

        start_time_series = self.pa.get_time_series()

        preproc = pre_processor(
            time_factor=time_factor, detector_factor=detector_factor
        )

        new_t, d, _ = preproc.run(start_time_series, self.pa)

        self.assertIsInstance(d, dict)
        self.assertIsInstance(new_t, PATimeSeries)
        self.assertEqual(
            new_t.shape,
            start_time_series.shape[:-2]
            + (
                start_time_series.shape[-2] * detector_factor,
                start_time_series.shape[-1] * time_factor,
            ),
        )
        self.assertTrue(np.all(np.isclose(detectors_start[0], d["geometry"][0])))
        return new_t

    def test_gpu_processing(self):
        if find_spec("cupy") is None:
            return None
        new_t = self._test_preprocessor(GPUMSOTPreProcessor)
        print(np.mean(new_t[0, 0].values))

    def test_numpy_processing(self):
        new_t = self._test_preprocessor(NumpyPreProcessor)
        print(np.mean(new_t[0, 0].values))

    def test_jax_preprocessing(self):
        new_t = self._test_preprocessor(PreProcessor)
        print(np.mean(new_t[0, 0].values))
