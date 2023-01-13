#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

"""
``msot-convert-simpa``: A script to convert simpa HDF5 output into the PATATO hdf5 format.
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
    parser.add_argument('-u', '--update', type=strtobool, default=False, help="Update metadata")
    parser.add_argument('-n', '--name', type=strtobool, default=False, help="Keep the user defined scan name and do "
                                                                            "not use the iThera defined generic name.")
    parser.add_argument('-s', '--slice', type=str, default="", help="Slice Select (integer or \"middle\")")
    args = parser.parse_args()

    try:
        slice_n = int(args.slice)
    except ValueError:
        slice_n = args.slice

    convert_simpa(args.input[0], args.output[0], args.update, args.name, slice_n=slice_n)
