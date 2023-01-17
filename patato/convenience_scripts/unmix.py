#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

"""
``msot-unmix`` is a command line tool for unmixing MSOT data.
"""

import argparse
import glob
import json
from os.path import join

import numpy as np
from .. import PAData
from .. import get_default_unmixing_preset
from .. import read_unmixing_preset
from .. import run_pipeline
from .. import sort_key


def init_argparse():
    parser = argparse.ArgumentParser(description="Process MSOT Data.")
    parser.add_argument('input', type=str, help="Data File")
    parser.add_argument('-w', '--wavelengths', nargs="*", type=float, default=[])
    parser.add_argument('-i', '--wavelengthindices', nargs="*", type=int, default=[])
    parser.add_argument('-p', '--preset', type=str, help="Preset File",
                        default=None)
    parser.add_argument('-c', '--clear', type=bool, help="Clear Old",
                        default=False)
    parser.add_argument('-f', '--filter', type=str, help="Scan Name Filter",
                        default=None)
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    preset = args.preset
    if preset is None:
        preset = get_default_unmixing_preset()

    DATA_FOLDER = args.input
    print(DATA_FOLDER)
    # Load json
    with open(preset) as json_file:
        settings = json.load(json_file)

    for data_file in sorted(glob.glob(join(DATA_FOLDER, "**", "*.hdf5"), recursive=True), key=sort_key):
        print("Processing", data_file)
        data = PAData.from_hdf5(data_file, "r+")

        if np.any(data.get_wavelengths() == 0.):
            print("Strange, some wavelengths = 0 - investigate.", data_file)
            continue

        if not data.get_recon_types():
            print(f"{data_file} has no reconstructions.")

        pipeline = read_unmixing_preset(settings, data)
        if args.filter is not None:
            if str.lower(args.filter) not in str.lower(data.get_scan_name()):
                print(data.get_scan_name(), "skipped")
                continue

        if args.clear:
            # Clear old processing data and processed
            data.delete_recons(recon_groups=["so2", "unmixed", "thb"])

        reconstructions = data.get_scan_reconstructions()
        for recon in data.get_scan_reconstructions():
            current_data = reconstructions[recon]
            run_pipeline(pipeline, current_data, data, n_batch=-1)


if __name__ == "__main__":
    main()
