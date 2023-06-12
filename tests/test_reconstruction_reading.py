#  Copyright (c) Thomas Else 2023.
#  License: MIT

import unittest
from os.path import join, split

from patato import SpectralUnmixer, PreProcessor
from patato.io.json.json_reading import read_reconstruction_preset, read_unmixing_preset
from patato.processing.preprocessing_algorithm import NumpyPreProcessor
from patato.processing.gpu_preprocessing_algorithm import GPUMSOTPreProcessor


class TestJSONLoading(unittest.TestCase):
    def test_load_recon_preset(self):
        folder, _ = split(__file__)
        pipeline = read_reconstruction_preset(join(folder,
                                                   "../patato/recon/recon_presets/backproject_standard_xz.json"))

        self.assertIn(type(pipeline), [PreProcessor, NumpyPreProcessor, GPUMSOTPreProcessor])

    def test_load_unmixed_preset(self):
        folder, _ = split(__file__)
        pipeline = read_unmixing_preset(join(folder, "../patato/unmixing/unmix_presets/haemoglobin.json"), None,
                                        WAVELENGTHS=[700, 900])

        self.assertEqual(type(pipeline), SpectralUnmixer)


if __name__ == '__main__':
    unittest.main()
