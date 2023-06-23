#  Copyright (c) Thomas Else 2023.
#  License: MIT

import unittest
from pathlib import Path
from unittest import mock
import argparse


class TestCommandLine(unittest.TestCase):
    def setUp(self) -> None:
        self.hdf5_file = "test_pipeline_folder"
        f = Path(self.hdf5_file)
        if not f.exists():
            f.mkdir()

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input="test_data_ithera/Scan_9",
                                                output="test_pipeline_folder"))
    def test_import_ithera(self, mock_args):
        from patato.convenience_scripts.convert_binary_to_hdf5 import main
        main()

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input=".",
                                                c=1500,
                                                clear=True,
                                                console=True))
    def test_tune_speed_of_sound(self, mock_args):
        from patato.convenience_scripts.tune_speed_of_sound import main
        main()

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input=".",
                                                clear=True,))
    def test_process_msot(self, mock_args):
        from patato.convenience_scripts.process_msot import main
        main()

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(input=".",
                                                clear=True, ))
    def test_unmix(self, mock_args):
        from patato.convenience_scripts.unmix import main
        main()
