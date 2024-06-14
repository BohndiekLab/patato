#  Copyright (c) Thomas Else 2023.
#  License: MIT

"""
``patato-unmix`` is a command line tool for unmixing MSOT data.
"""

import argparse
import glob
import os.path
from os.path import join

import numpy as np
from .. import PAData
from ..unmixing.learned_unmixer import LearnedSpectralUnmixer
from ..utils import sort_key
from ..utils.pipeline import run_pipeline


def init_argparse():
    parser = argparse.ArgumentParser(description="Process MSOT Data.")
    parser.add_argument('input', type=str, help="Data File")
    parser.add_argument('-w', '--wavelengths', nargs="*", type=float, default=[])
    parser.add_argument('-t', '--traindata', type=str, help="Training data set to use, e.g. 'BASE'", default=None)
    parser.add_argument('-r', '--resample', type=int, help="Downsampling factor", default=None)
    parser.add_argument('-c', '--clear', type=bool, help="Clear Old", default=False)
    parser.add_argument('-f', '--filter', type=str, help="Scan Name Filter", default=None)
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    traindata = args.traindata
    if traindata is None:
        traindata = "BASE"

    rescaling_factor = args.resample
    if rescaling_factor is None:
        rescaling_factor = 1

    wavelengths = args.wavelengths
    if wavelengths is None:
        wavelengths = []

    data_folder = args.input
    print(data_folder)

    if os.path.isfile(data_folder):
        data_files = [data_folder]
    else:
        data_files = sorted(glob.glob(join(data_folder, "**", "*.hdf5"), recursive=True), key=sort_key)

    for data_file in data_files:
        print("Processing", data_file)
        data = PAData.from_hdf5(data_file, "r+")

        if wavelengths is None or len(wavelengths) == 0:
            wavelengths = data.get_wavelengths()

        if np.any(data.get_wavelengths() == 0.):
            print("Strange, some wavelengths = 0 - investigate.", data_file)
            continue

        if not data.get_recon_types():
            print(f"{data_file} has no reconstructions.")

        pipeline = LearnedSpectralUnmixer(train_dataset_id=traindata,
                                          wavelengths=wavelengths,
                                          rescaling_factor=rescaling_factor)
        if args.filter is not None:
            if str.lower(args.filter) not in str.lower(data.get_scan_name()):
                print(data.get_scan_name(), "skipped")
                continue

        if args.clear:
            # Clear old processing data and processed
            data.delete_recons(recon_groups=["learned_sO2"])

        reconstructions = data.get_scan_reconstructions()
        for recon in data.get_scan_reconstructions():
            current_data = reconstructions[recon]
            run_pipeline(pipeline, current_data, data, n_batch=-1)


if __name__ == "__main__":
    main()
