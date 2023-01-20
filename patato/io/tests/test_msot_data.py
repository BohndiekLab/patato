#  Copyright (c) Thomas Else 2023.
#  License: BSD-3
import unittest
from os.path import join

import numpy as np

from ...data.get_example_datasets import get_patato_data_folder, get_msot_time_series_example
from ..msot_data import PAData


class TestMSOTData(unittest.TestCase):
    def test_msot_data(self):
        pa = get_msot_time_series_example("so2")[:20]
        # print(pa.get_scan_name(), str(pa.get_scan_datetime()))
        pa.set_default_recon()
        self.assertTrue(pa.get_scan_name() == "201001_05_GC_SS")
        self.assertTrue(str(pa.get_scan_datetime()) == "2020-10-01 12:32:32.629145")
        self.assertEqual(pa.get_n_samples(), 2030)
        self.assertEqual(pa.get_scan_reconstructions().raw_data.shape, (20, 10, 1, 333, 333))
        self.assertEqual(pa.get_scan_unmixed().raw_data.shape, (20, 2, 1, 111, 111))
        self.assertEqual(pa.get_scan_so2().raw_data.shape, (20, 1, 1, 111, 111))
        self.assertEqual(pa.get_sampling_frequency(), 4e7)
        self.assertEqual(pa.get_scan_images("not a real image"), {})
        self.assertTrue(np.all(np.squeeze(pa.get_scan_so2_time_mean().raw_data) == np.squeeze(np.mean(pa.get_scan_so2().raw_data, axis=0))))
        self.assertTrue(np.all(np.squeeze(pa.get_scan_thb_time_mean().raw_data) == np.squeeze(np.mean(pa.get_scan_thb().raw_data, axis=0))))
        pa.close()


if __name__ == "__main__":
    t = TestMSOTData()
    t.test_msot_data()
