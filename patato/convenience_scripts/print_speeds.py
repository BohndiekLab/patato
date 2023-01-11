#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse
import glob
from os.path import join, split

import h5py
from .. import sort_key
from tabulate import tabulate


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

    names = []
    cs = []
    fnames = []
    print(args.input)

    for filename in sorted(glob.glob(join(args.input, "*.hdf5")), key=sort_key):
        data_file = h5py.File(filename, "r")

        # 1. Check raw data and print name

        if "raw_data" in data_file:
            fname = split(filename)[-1]
            name = data_file["raw_data"].attrs["name"]
            sos = str(data_file["raw_data"].attrs.get("speedofsound", "NOT SET"))
            if len(names) >= 1:
                if names[-1].split("-")[0].split("_")[0] != name.split("-")[0].split("_")[0] and names[-1][0] != "=":
                    names.append("==========")
                    cs.append("==========")
                    fnames.append("=========")
            names.append(name)
            fnames.append(fname)
            cs.append(sos)

    print(tabulate(zip(fnames, names, cs), headers=["File Name", "Scan Name", "Speed of Sound"]))
