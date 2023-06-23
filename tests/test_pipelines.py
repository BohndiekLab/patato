#  Copyright (c) Thomas Else 2023.
#  License: MIT

import unittest
from os.path import join

from patato import PreProcessor
from patato.core.image_structures.reconstruction_image import Reconstruction
from patato.core.image_structures.unmixed_image import UnmixedData
from patato.data.get_example_datasets import get_patato_data_folder
from patato.io.msot_data import PAData
from patato.unmixing.spectra import OxyHaemoglobin, Haemoglobin
from patato.recon import ReferenceBackprojection
from patato.unmixing.unmixer import SpectralUnmixer
from patato.utils.pipeline import run_pipeline

from make_dummy_dataset import make_dummy_dataset


class TestPipelines(unittest.TestCase):
    def setUp(self) -> None:
        make_dummy_dataset()

    def test_full_run(self):
        data_folder = join(get_patato_data_folder(), "test")
        dummy_dataset = join(data_folder, "Scan_1.hdf5")

        pa = PAData.from_hdf5(dummy_dataset, "r+")[:, 0:2]

        preproc = PreProcessor(time_factor=3, detector_factor=2)
        reconstructor = ReferenceBackprojection([333, 1, 333], [0.025, 1, 0.025])
        unmixer = SpectralUnmixer([OxyHaemoglobin(), Haemoglobin()], pa.get_wavelengths())

        preproc.add_child(reconstructor)
        reconstructor.add_child(unmixer)

        results, additional_results = run_pipeline(preproc, pa.get_time_series(), pa, n_batch=-1, save_results=False)

        self.assertEqual(len(additional_results), 0)
        self.assertEqual(type(results[0]), Reconstruction)
        self.assertEqual(type(results[1]), UnmixedData)