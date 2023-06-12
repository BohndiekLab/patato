#  Copyright (c) Thomas Else 2023.
#  License: MIT
import unittest

import patato as pat

import numpy as np
from patato.data import get_msot_time_series_example


class TestModelBased(unittest.TestCase):
    def setUp(self) -> None:
        self.pa = get_msot_time_series_example("so2")[0:1, 0:1]

    def test_model_based_reconstruction(self):
        N, fov = (100, 100, 1), (0.025, 0.025, 1)
        mb = pat.ModelBasedReconstruction(N, fov, pa_example=self.pa)
        rec, _, _ = mb.run(self.pa.get_time_series(), self.pa)
        self.assertAlmostEqual(np.mean(rec.values), 15.583445802192289, 2)
        self.assertEqual(rec.shape, (1, 1, 1, 100, 100))
