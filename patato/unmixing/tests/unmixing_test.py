#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import unittest
from os.path import join, split

import numpy as np

from ... import Reconstruction, SO2Calculator, SpectralUnmixer, THbCalculator, ReferenceBackprojection
from ...io.msot_data import PAData
from ...processing.preprocessing_algorithm import DefaultMSOTPreProcessor
from ...data.get_example_datasets import get_patato_data_folder, get_msot_time_series_example


class TestUnmixing(unittest.TestCase):
    def test_numpy_unmix(self):
        image = np.zeros((1, 2, 333, 333, 1))
        image[:, 0] += 1112 # Manually chosen values to match the absorption of 650 nm and 800 nm for 78% oxygenated Hb
        image[:, 1] += 804
        wavelengths = np.array([650, 800])

        r = Reconstruction(image, wavelengths,
                           field_of_view=(1, 1, 1))  # field of view is the width of the image along x, y, z

        r.attributes["RECONSTRUCTION_FIELD_OF_VIEW_X"] = 1
        r.attributes["RECONSTRUCTION_FIELD_OF_VIEW_Y"] = 1
        r.attributes["RECONSTRUCTION_FIELD_OF_VIEW_Z"] = 1

        um = SpectralUnmixer(["Hb", "HbO2"], r.wavelengths)
        so = SO2Calculator()

        um, _, _ = um.run(r, None)
        so2, _, _ = so.run(um, None)

        self.assertAlmostEqual(np.mean(so2.raw_data), 0.7799958428309177)

    def test_unmix(self):
        pa = get_msot_time_series_example("so2")[:3]

        pa.set_default_recon()
        r = pa.get_scan_reconstructions()

        unmixer = SpectralUnmixer(["Hb", "HbO2"], pa.get_wavelengths())

        u, _, _ = unmixer.run(r, pa)

        so2_calc = SO2Calculator()
        s, _, _ = so2_calc.run(u, pa)

        thb_calc = THbCalculator()
        t, _, _ = thb_calc.run(u, pa)
        self.assertTrue(np.all(pa.get_scan_so2_time_mean().raw_data == np.mean(pa.get_scan_so2().raw_data,
                                                                               axis=(0, 1))))
        self.assertTrue(np.all(pa.get_scan_so2_time_standard_deviation().raw_data == np.std(pa.get_scan_so2().raw_data,
                                                                                            axis=(0, 1))))

        # Get these to do sanity checks - better to implement proper tests in future.
        self.assertIsNotNone(pa.get_scan_so2_frequency_components(fnum=3))
        self.assertIsNotNone(pa.get_scan_so2_frequency_peak(fnum=3))
        self.assertIsNotNone(pa.get_scan_so2_frequency_sum(fnum=3))
        self.assertIsNone(pa.get_segmentation())
        self.assertIsNotNone(pa.get_recon_types())
        self.assertIsNotNone(pa.get_fft())
        self.assertIsNotNone(pa.get_recon_types())
        self.assertIsNotNone(pa.get_scan_dso2())
        self.assertIsNotNone(pa.get_scan_baseline_so2())
        self.assertIsNotNone(pa.get_scan_baseline_standard_deviation_so2())
        self.assertIsNotNone(pa.get_scan_dicg())
        self.assertIsNotNone(pa.get_scan_baseline_icg())
        self.assertIsNotNone(pa.get_scan_baseline_standard_deviation_icg())
        self.assertIsNone(pa.get_responding_pixels())
        self.assertIsNotNone(pa.get_rois())
        self.assertIsNotNone(pa.summary_measurements())

        self.assertEqual(u.shape, (3, 2, 1, 333, 333))
        self.assertEqual(s.shape, (3, 1, 1, 333, 333))
        self.assertEqual(t.shape, (3, 1, 1, 333, 333))
