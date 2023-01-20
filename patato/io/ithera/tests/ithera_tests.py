#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import unittest

import numpy as np
from .... import DefaultMSOTPreProcessor, ReferenceBackprojection
from ....data import get_ithera_msot_time_series_example, get_msot_time_series_example
import matplotlib.pyplot as plt


class TestITheraImport(unittest.TestCase):
    def test_overall_processing(self):
        pa_1 = get_ithera_msot_time_series_example("so2")
        pa_2 = get_msot_time_series_example("so2")

        self.assertEqual(pa_1.get_scan_name(), pa_2.get_scan_name())
        self.assertEqual(pa_1.scan_reader.get_scan_comment(), pa_2.scan_reader.get_scan_comment())

        self.assertEqual(pa_1.get_scan_reconstructions(), {})
        pa_2.set_default_recon()

        pa_2.get_scan_reconstructions().imshow(return_scalebar_dimension=True)
        plt.close()

        us = pa_2.scan_reader.get_us_data()
        self.assertIsNone(us[0])

        time_factor = 3
        detector_factor = 2

        preproc = DefaultMSOTPreProcessor(time_factor=time_factor, detector_factor=detector_factor)
        recon = ReferenceBackprojection(field_of_view=(0.025, 0.025, 0.025), n_pixels=(333, 333, 1))

        new_t1, d1, _ = preproc.run(pa_1.get_time_series()[0:1, 0:1], pa_1[0:1, 0:1])
        rec1, _, _ = recon.run(new_t1, pa_1[0:1, 0:1], 1500, **d1)

        new_t2, d2, _ = preproc.run(pa_2.get_time_series()[0:1, 0:1], pa_2[0:1, 0:1])
        rec2, _, _ = recon.run(new_t2, pa_2[0:1, 0:1], 1500, **d2)

        self.assertTrue(np.all(pa_1.get_time_series().raw_data[()] == pa_2.get_time_series().raw_data[()]))
        self.assertTrue(np.all(new_t1.raw_data[()] == new_t2.raw_data[()]))
