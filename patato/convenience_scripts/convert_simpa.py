#  Copyright (c) Thomas Else 2023.
#  License: MIT

"""
``patato-convert-simpa``: A script to convert simpa HDF5 output into the PATATO hdf5 format.
"""

import argparse
from distutils.util import strtobool

from ..io.hdf5_converter import convert_simpa


def main():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-hv] input output",
        description="Convert Simpa Data into a standard hdf5 format. ."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1"
    )
    parser.add_argument('input', nargs=1, type=str, help="Simpa Study Folder")
    parser.add_argument('output', nargs=1, help="Output Folder")
    parser.add_argument('-n', '--name', type=strtobool, default=False, help="Keep the user defined scan name and do "
                                                                            "not use the iThera defined generic name.")
    args = parser.parse_args()

    convert_simpa(args.input[0], args.output[0], args.name)
