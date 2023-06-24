#  Copyright (c) Thomas Else 2023.
#  License: MIT

import unittest

import numpy as np

from patato import PAData
from patato.processing.jax_preprocessing_algorithm import PreProcessor
from patato.recon import OpenCLBackprojection, ReferenceBackprojection, SlowBackprojection


class TestBackprojection(unittest.TestCase):
    def setUp(self) -> None:
        self.pa = PAData.from_hdf5("test_data.hdf5")

        self.preproc = PreProcessor(time_factor=1, detector_factor=2)
        self.filtered_time_series, self.new_settings, _ = self.preproc.run(self.pa.get_time_series(), self.pa)

    def _test_backprojector(self, reconstructor_class):
        reconstructor = reconstructor_class([333, 334, 1], [0.025, 0.025, 1.])
        r, _, _ = reconstructor.run(self.filtered_time_series, self.pa, **self.new_settings)
        self.assertEqual(r.shape, (1, 2, 1, 334, 333))
        self.assertAlmostEqual(np.mean(r[0, 0].values), 315.63669659956736, 2)

        reconstructor = reconstructor_class([1, 333, 334], [1., 0.025, 0.025])
        r, _, _ = reconstructor.run(self.filtered_time_series, self.pa, **self.new_settings)
        self.assertEqual(r.shape, (1, 2, 334, 333, 1))

        reconstructor = reconstructor_class([334, 1, 333], [0.025, 1., 0.025])
        r, _, _ = reconstructor.run(self.filtered_time_series, self.pa, **self.new_settings)
        self.assertEqual(r.shape, (1, 2, 333, 1, 334))

    def test_reference_reconstruction(self):
        self._test_backprojector(ReferenceBackprojection)

    def test_slow_backprojection(self):
        self._test_backprojector(SlowBackprojection)

    def test_opencl_reconstruction(self):
        try:
            import pyopencl
        except ImportError:
            return  # Skip test if pyopencl is not installed

        self._test_backprojector(OpenCLBackprojection)
