#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import unittest
from os.path import join, split

from ...core.image_structures.reconstruction_image import Reconstruction
from ...core.image_structures.unmixed_image import UnmixedData
from ...data.get_example_datasets import get_patato_data_folder
from ...io.msot_data import PAData
from ...processing.preprocessing_algorithm import DefaultMSOTPreProcessor
from ...unmixing.spectra import OxyHaemoglobin, Haemoglobin
from ...recon import ReferenceBackprojection
from ...unmixing.unmixer import SpectralUnmixer
from ..run_pipeline import run_pipeline


class TestPipelines(unittest.TestCase):
    def test_full_run(self):
        data_folder = join(get_patato_data_folder(), "test")
        dummy_dataset = join(data_folder, "Scan_1.hdf5")

        pa = PAData.from_hdf5(dummy_dataset, "r+")[:, 0:2]

        preproc = DefaultMSOTPreProcessor(time_factor=3, detector_factor=2)
        reconstructor = ReferenceBackprojection([333, 1, 333], [0.025, 1, 0.025])
        unmixer = SpectralUnmixer([OxyHaemoglobin(), Haemoglobin()], pa.get_wavelengths())

        preproc.add_child(reconstructor)
        reconstructor.add_child(unmixer)

        results, additional_results = run_pipeline(preproc, pa.get_time_series(), pa, n_batch=-1, save_results=False)

        self.assertEqual(len(additional_results), 0)
        self.assertEqual(type(results[0]), Reconstruction)
        self.assertEqual(type(results[1]), UnmixedData)