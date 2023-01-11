#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse
import glob
from os import makedirs, system
from os.path import join, split, exists


def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-hv] input output",
        description="Convert iThera MSOT Data into a hdf5 format. ."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1"
    )
    parser.add_argument('input', type=str, help="iThera Studies Folder")
    parser.add_argument('output', help="Empty Output Folder")
    parser.add_argument('-u', '--update', type=bool, default=False, help="Update metadata")
    parser.add_argument('-g', '--dontgetrecons', type=bool, default=False, help="Don't get Recons")
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    for folder in glob.glob(join(args.input, "Study_*")):
        study_name = split(folder)[-1]
        print(f"-----{study_name}-----")
        study_output = join(args.output, study_name)
        if not exists(study_output):
            makedirs(study_output)
        system(f"msot-import-ithera \"{folder}\" \"{study_output}\"" + (
            " -g True" if args.dontgetrecons else ""))
