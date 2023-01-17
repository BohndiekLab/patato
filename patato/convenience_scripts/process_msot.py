#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

"""
``msot-reconstruct``: A script to reconstruct all MSOT data files in a folder.
"""

import argparse
import glob
import json
import logging
import os

from .. import PAData
from .. import get_default_recon_preset
from .. import read_reconstruction_preset
from .. import run_pipeline
from .. import sort_key


def main():
    parser = argparse.ArgumentParser(description="Process MSOT Data.")
    parser.add_argument('input', type=str, help="Data File")
    parser.add_argument('-c', '--speed', type=float, help="Speed of Sound", default=None)
    parser.add_argument('-o', '--output', type=str, help="Output File", default=None)
    parser.add_argument('-p', '--preset', type=str, help="Preset File",
                        default=None)
    parser.add_argument('-r', '--run', type=int, help="Run Number",
                        default=None)
    parser.add_argument('-w', '--wavelength', type=int, help="Wavelength Number",
                        default=None)
    parser.add_argument('-R', '--repeat', type=lambda x: x in ["True", "true", "T", "Y", "y", "yes", "YES", "1"],
                        default=True, help="Repeat recon if already exists.")
    parser.add_argument('-H', '--highpass', type=float, help="High pass filter override", default=None)
    parser.add_argument('-L', '--lowpass', type=float, help="Low pass filter override", default=None)
    parser.add_argument('-C', '--clear', type=bool, default=False, help="Clear all existing recons.")
    parser.add_argument('-I', '--ipasc', type=bool, default=False, help="Export to IPASC format as well.")
    parser.add_argument('-d', '--debug', type=bool, default=False, help="Enable debugging logging.")
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    files = []

    data_file = args.input
    if os.path.isfile(data_file):
        files.append(data_file)
    elif os.path.isdir(data_file):
        files = sorted(glob.glob(os.path.join(data_file, "**", "*.hdf5"), recursive=True), key=sort_key)

    output_file = args.output

    clear_recons = args.clear
    recompute_recon = args.repeat
    speed_of_sound = args.speed
    run_number = args.run
    if run_number is None:
        run_number = slice(None, None)
    else:
        run_number = slice(run_number, run_number + 1)
    wavelength = args.wavelength
    if wavelength is None:
        wavelength = slice(None, None)
    else:
        wavelength = slice(wavelength, wavelength + 1)
    do_ipasc_export = args.ipasc

    print(f"Identified {len(files)} files to process!")

    for data_file in files:
        # Run processing.
        pa_data = PAData.from_hdf5(data_file, "r+")[run_number, wavelength]

        json_path = args.preset
        if json_path is None:
            json_path = get_default_recon_preset(pa_data)

        with open(json_path) as json_file:
            settings = json.load(json_file)

        if args.lowpass is not None:
            settings["FILTER_LOW_PASS"] = args.lowpass
        elif args.highpass is not None:
            settings["FILTER_HIGH_PASS"] = args.highpass

        # Setup the reconstruction pipeline
        pipeline = read_reconstruction_preset(settings)

        # Set the reconstruction description
        description = pipeline.children[0].get_algorithm_name()

        # Setup the name
        if args.highpass is not None:
            description += "_CUSTOM_HP_FILTER"
        if args.lowpass is not None:
            description += "_CUSTOM_LP_FILTER"
        if args.run is not None:
            description += "_run_" + str(args.run)
        if args.wavelength is not None:
            description += "_wavelength_" + str(args.run)

        # Start processing
        if clear_recons:
            pa_data.delete_recons()

        if description in pa_data.get_recon_types() and not recompute_recon:
            print("Already computed recon", description, "for", data_file + ".")
            continue

        print(data_file, pa_data.dataset.shape)
        if any(x == 0 for x in pa_data.dataset.shape):
            continue

        run_pipeline(pipeline, pa_data.dataset, pa_data, n_batch=1 + 20 // pa_data.shape[1], output_file=output_file)

        if do_ipasc_export:
            from .. import export_to_ipasc
            export_to_ipasc(data_file)


if __name__ == "__main__":
    main()
