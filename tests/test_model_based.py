#  Copyright (c) Thomas Else 2023.
#  License: MIT
import unittest

import patato as pat

import numpy as np

from patato import PAData


class TestModelBased(unittest.TestCase):
    def setUp(self) -> None:
        self.pa = PAData.from_hdf5("test_data.hdf5")

    def test_model_based_reconstruction(self):
        N, fov = (100, 100, 1), (0.025, 0.025, 1)
        mb = pat.ModelBasedReconstruction(N, fov, pa_example=self.pa)
        rec, _, _ = mb.run(self.pa.get_time_series(), self.pa)
        self.assertAlmostEqual(np.mean(rec.values), 15.583445802192289, 2)
        self.assertEqual(rec.shape, (1, 1, 1, 100, 100))
