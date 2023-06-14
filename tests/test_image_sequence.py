#  Copyright (c) Thomas Else 2023.
#  License: MIT

import tempfile
import unittest

import h5py
import numpy as np

from patato import PAData


class TestHDF5Load(unittest.TestCase):
    def setUp(self) -> None:
        tf = tempfile.TemporaryFile()
        self.file = h5py.File(tf, "a")

        # Add some raw_data
        raw_data = np.zeros((1, 11, 256, 2030))
        self.file.create_dataset("raw_data", data=raw_data)

        # Add some wavelengths
        self.file.create_dataset("wavelengths", data=np.linspace(700, 900, 11))

        # Impulse response
        self.file.create_dataset("irf", data=np.ones(2030))

        # GEOMETRY
        self.file.create_dataset("GEOMETRY", data=np.ones((256, 3)))

        # Reconstructions
        self.file.require_group("recons")
        self.file.require_group("test_recon")
        self.file.create_dataset("recons/test_recon/0", data=np.ones((1, 11, 333, 333, 1)))
        self.file["recons/test_recon/0"].attrs["RECONSTRUCTION_FIELD_OF_VIEW_X"] = 0.025
        self.file["recons/test_recon/0"].attrs["RECONSTRUCTION_FIELD_OF_VIEW_Y"] = 0.025
        self.file["recons/test_recon/0"].attrs["RECONSTRUCTION_FIELD_OF_VIEW_Z"] = 0.
        self.file["recons/test_recon/0"].attrs["RECONSTRUCTION_NX"] = 333
        self.file["recons/test_recon/0"].attrs["RECONSTRUCTION_NY"] = 333
        self.file["recons/test_recon/0"].attrs["RECONSTRUCTION_NZ"] = 1
        self.pa_data = PAData.from_hdf5(self.file)

    def test_from_hdf5_dataset(self):
        r = self.pa_data.get_scan_reconstructions()
        self.assertEqual(len(r), 1)
        self.assertEqual(list(r.keys())[0], ("test_recon", "0"))
        self.assertEqual(self.pa_data.get_scan_reconstructions()[("test_recon", "0")].get_ax1_label_meaning(),
                         "wavelengths")

    def test_slicing(self):
        data = self.pa_data.get_scan_reconstructions()["test_recon", "0"]
        self.assertEqual(data.shape, (1, 11, 333, 333, 1))
        self.assertEqual(data[0].shape, (11, 333, 333, 1))
        self.assertEqual(data[:, 0].shape, (1, 333, 333, 1))

        self.assertTrue(np.all(data[0].ax_1_labels == np.linspace(700, 900, 11)))
        self.assertEqual(data[0, 0].ax_1_labels.size, 1)
        for i in range(11):
            self.assertEqual(data[0, i].ax_1_labels.item(), np.linspace(700, 900, 11)[i])
        self.assertEqual(data[0].cmap, "bone")
        self.assertEqual(data[0].two_dims(), ("y", "z"))

        with self.assertRaises(IndexError):
            print(data[1])
            print(data[-2])
        #
