#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

"""
``msot-import-ithera``: A script to convert iThera MSOT Data into the PATATO hdf5 format.
"""

import argparse
from distutils.util import strtobool

from ..io.hdf5_converter import convert_ithera_msot_binary_to_hdf5


def main():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-hv] input output",
        description="Convert iThera MSOT Data into a hdf5 format. ."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1"
    )
    parser.add_argument('input', nargs=1, type=str, help="iThera Study Folder")
    parser.add_argument('output', nargs=1, help="Output Folder")
    parser.add_argument('-u', '--update', type=strtobool, default=False, help="Update metadata")
    parser.add_argument('-n', '--name', type=strtobool, default=False, help="Keep the user defined scan name and do "
                                                                            "not use the iThera defined generic name.")
    args = parser.parse_args()

    convert_ithera_msot_binary_to_hdf5(args.input[0], args.output[0], args.update, args.name)


if __name__ == "__main__":
    main()
