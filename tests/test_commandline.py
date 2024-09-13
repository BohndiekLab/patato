#  Copyright (c) Thomas Else 2023.
#  License: MIT

import unittest
from pathlib import Path
from unittest import mock
import argparse

from patato.convenience_scripts.convert_binary_to_hdf5 import main as main_convert
from patato.convenience_scripts.process_msot import main as main_process_msot
from patato.convenience_scripts.unmix import main as main_unmix


class TestCommandLine(unittest.TestCase):
    def setUp(self) -> None:
        self.hdf5_file = "test_pipeline_folder"
        f = Path(self.hdf5_file)
        if not f.exists():
            f.mkdir()

    @mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            input="test_data_ithera/Scan_9",
            output="test_pipeline_folder",
            update=False,
            name=False,
        ),
    )
    def test_import_ithera(self, mock_args):

        main_convert()

    @mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            input=".",
            speed=None,
            output=None,
            preset=None,
            run=None,
            wavelength=None,
            repeat=True,
            highpass=None,
            lowpass=None,
            clear=False,
            ipasc=False,
            debug=False,
        ),
    )
    def test_process_msot(self, mock_args):

        main_process_msot()

    @mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            input=".",
            wavelength=[],
            wavelengthindices=[],
            preset=None,
            clear=True,
            filter=None,
        ),
    )
    def test_unmix(self, mock_args):

        main_unmix()
