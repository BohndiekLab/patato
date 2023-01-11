#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse
import glob
from os.path import join

import h5py
from .. import sort_key


def init_argparse():
    parser = argparse.ArgumentParser(description="Look at the status of a data folder.")
    parser.add_argument('input', type=str, help="Data Folder")
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1"
    )
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    for filename in sorted(glob.glob(join(args.input, "*.hdf5")), key=sort_key):
        print(filename)
        data_file = h5py.File(filename, "r")

        # 1. Check raw data and print name

        if "raw_data" in data_file:
            print("Scan name", data_file["raw_data"].attrs["name"])
        print("Raw data present" if "raw_data" in data_file else "Raw data not present")
        if "raw_data" in data_file:
            print("Speed of sound set" if "speedofsound" in data_file["raw_data"].attrs else "Speed of sound not set")

        # 2. Check reconstructions

        if "recons" in data_file:
            for method in data_file["recons"]:
                print(method + ": " + str(len(data_file["recons"][method])))
        else:
            print("No reconstructions present")

        # 3. Unmixing?

        if "unmixed" in data_file:
            for method in data_file["unmixed"]:
                print("Unmixed results:" + method + ": " + str(len(data_file["unmixed"][method])))
        else:
            print("No unmixed results present")

        # 4. Rois?

        if "rois" in data_file:
            for region in data_file["rois"]:
                print(region + ":", len(data_file["rois"][region]))
        else:
            print("No rois available")
        print("\n")
