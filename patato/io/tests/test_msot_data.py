#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import unittest
from os.path import split, join

import numpy as np
from ..msot_data import PAData


class TestMSOTData(unittest.TestCase):
    def test_msot_data(self):
        f = split(__file__)[0]
        pa = PAData.from_hdf5(join(f, "../../../data/Scan_1.hdf5"), "r+")[:, 0:2]

        pa.set_default_recon()
        self.assertTrue(pa.get_scan_name() == "Demo Data")
        self.assertTrue(str(pa.get_scan_datetime()) == "2022-11-23 15:50:22")
        self.assertEqual(pa.get_n_samples(), 2030)
        self.assertEqual(pa.get_scan_reconstructions().raw_data.shape, (20, 2, 333, 1, 333))
        self.assertEqual(pa.get_scan_unmixed().raw_data.shape, (20, 2, 333, 1, 333))
        self.assertEqual(pa.get_scan_so2().raw_data.shape, (20, 1, 333, 1, 333))
        self.assertEqual(pa.get_sampling_frequency(), 4e7)
        self.assertEqual(pa.get_scan_images("not a real image"), {})
        self.assertTrue(np.all(np.squeeze(pa.get_scan_so2_time_mean().raw_data) == np.squeeze(np.mean(pa.get_scan_so2().raw_data, axis=0))))
        self.assertTrue(np.all(np.squeeze(pa.get_scan_thb_time_mean().raw_data) == np.squeeze(np.mean(pa.get_scan_thb().raw_data, axis=0))))
        pa.close()


if __name__ == "__main__":
    t = TestMSOTData()
    t.test_msot_data()
