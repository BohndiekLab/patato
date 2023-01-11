#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import unittest
from os.path import split, join

from ...io.msot_data import PAData
from ...processing.preprocessing_algorithm import DefaultMSOTPreProcessor
from .. import OpenCLBackprojection
from ..backprojection_reference import ReferenceBackprojection


class BackprojectionTest(unittest.TestCase):
    def test_reference_reconstruction(self):
        f = split(__file__)[0]
        pa = PAData.from_hdf5(join(f, "../../../data/Scan_1.hdf5"))[0:1, 0:1]

        preproc = DefaultMSOTPreProcessor(time_factor=1, detector_factor=2)
        filtered_time_series, new_settings, _ = preproc.run(pa.get_time_series(), pa)

        reconstructor = ReferenceBackprojection([333, 1, 333], [0.025, 1, 0.025])

        r, _, _ = reconstructor.run(filtered_time_series, pa, **new_settings)

        self.assertEqual(r.shape, (1, 1, 333, 1, 333))
        # self.assertAlmostEqual(np.mean(r.values), 643.466579199574)

    def test_opencl_reconstruction(self):
        import jax
        if jax.devices()[0].platform == "cpu":
            # Don't test opencl reconstruction on a cpu - painfully slow.
            return None
        f = split(__file__)[0]
        pa = PAData.from_hdf5(join(f, "../../../data/Scan_1.hdf5"))[0:1, 0:1]

        preproc = DefaultMSOTPreProcessor(time_factor=1, detector_factor=2)
        filtered_time_series, new_settings, _ = preproc.run(pa.get_time_series(), pa)

        reconstructor = OpenCLBackprojection([333, 1, 333], [0.025, 1, 0.025])

        r, _, _ = reconstructor.run(filtered_time_series, pa, **new_settings)
        self.assertEqual(r.shape, (1, 1, 333, 1, 333))
        # self.assertAlmostEqual(np.mean(r.values), 643.4664306641)
