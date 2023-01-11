#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import unittest
from os.path import split, join

import numpy as np
from .... import PAData, iTheraMSOT, DefaultMSOTPreProcessor, ReferenceBackprojection


class TestITheraImport(unittest.TestCase):
    def test_overall_processing(self):
        f = split(__file__)[0]
        # old version
        pa_1 = PAData(iTheraMSOT(join(f, "../../../../data/itheraexample/Scan_1")))
        # New version
        pa_2 = PAData(iTheraMSOT(join(f, "../../../../data/itheraexample/Scan_1")))
        self.assertEqual(pa_1.get_scan_name(), pa_2.get_scan_name())
        self.assertEqual(pa_1.scan_reader.get_scan_comment(), pa_2.scan_reader.get_scan_comment())

        print(pa_1.get_scan_reconstructions())
        pa_2.set_default_recon()

        pa_2.get_scan_reconstructions().imshow(return_scalebar_dimension=True)

        us = pa_2.scan_reader.get_us_data()
        self.assertIsNone(us)
        time_factor = 3
        detector_factor = 2

        preproc = DefaultMSOTPreProcessor(time_factor=time_factor, detector_factor=detector_factor)
        recon = ReferenceBackprojection(field_of_view=(0.025, 0.025, 0.025), n_pixels=(333, 333, 1))

        new_t1, d1, _ = preproc.run(pa_1.get_time_series(), pa_1)
        rec1, _, _ = recon.run(new_t1, pa_1, 1500, **d1)

        new_t2, d2, _ = preproc.run(pa_2.get_time_series(), pa_2)
        rec2, _, _ = recon.run(new_t2, pa_2, 1500, **d2)

        self.assertTrue(np.all(pa_1.get_time_series().raw_data[()] == pa_2.get_time_series().raw_data[()]))
        self.assertTrue(np.all(new_t1.raw_data[()] == new_t2.raw_data[()]))
