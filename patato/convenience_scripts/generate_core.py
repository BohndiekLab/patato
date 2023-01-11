#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse
import glob
from os.path import split, join

from .. import PAData
from .. import sort_key
from ..utils.roi_operations import add_rim_core_data


def init_argparse():
    parser = argparse.ArgumentParser(description="Split regions of interest.")
    parser.add_argument('input', type=str, help="Data Folder")
    parser.add_argument('name', type=str, help="ROI Name")
    parser.add_argument('number', type=str, help="ROI Number")
    parser.add_argument('distance', type=float, help="Core Distance (in metres)")
    parser.add_argument('-r', '--radius', type=bool, help="Distance is desired radius", default=False)
    parser.add_argument('-f', '--filter', type=str, help="Choose scan", default=None)
    parser.add_argument('-fn', '--filtername', type=str, help="Choose scan name filter", default=None)
    parser.add_argument('-c', '--clear', type=str, help="Clear all generated ROIs", default=False)
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    data_folder = args.input

    for file in sorted(glob.glob(join(data_folder, "**", "*.hdf5"),recursive=True), key=sort_key):
        if args.filter is not None:
            if split(file)[-1] != "Scan_" + args.filter + ".hdf5":
                continue
        data = PAData.from_hdf5(file, "r+")

        if args.filtername is not None:
            if args.filtername.upper() not in data.get_scan_name().upper():
                continue

        if args.clear and "rois" in data.file:
            for name in data.file["rois"]:
                for num in data.file["rois"][name]:
                    if data.file["rois"][name][num].attrs.get("generated", False):
                        del data.file["rois"][name][num]

        if (args.name, args.number) in data.get_rois():
            print(file)
            new_roi, _ = add_rim_core_data(data, (args.name, args.number), args.distance, radius=args.radius)
            data.add_roi(new_roi, generated=True)
            data.file.close()


if __name__ == "__main__":
    main()
