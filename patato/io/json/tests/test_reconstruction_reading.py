#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import unittest
from os.path import join, split

from .... import SpectralUnmixer, MSOTPreProcessor
from ....io.json.json_reading import read_reconstruction_preset, read_unmixing_preset
from ....processing.preprocessing_algorithm import DefaultMSOTPreProcessor
from ....processing.gpu_preprocessing_algorithm import GPUMSOTPreProcessor


class TestJSONLoading(unittest.TestCase):
    def test_load_recon_preset(self):
        folder, _ = split(__file__)
        pipeline = read_reconstruction_preset(join(folder, "../../../recon/recon_presets/backproject_standard_xz.json"))

        self.assertIn(type(pipeline), [MSOTPreProcessor, DefaultMSOTPreProcessor, GPUMSOTPreProcessor])

    def test_load_unmixed_preset(self):
        folder, _ = split(__file__)
        pipeline = read_unmixing_preset(join(folder, "../../../unmixing/unmix_presets/haemoglobin.json"), None,
                                        WAVELENGTHS=[700, 900])

        self.assertEqual(type(pipeline), SpectralUnmixer)


if __name__ == '__main__':
    unittest.main()
