#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import os
import glob
from os.path import join, dirname, split

from ..io.ithera.read_ithera import iTheraMSOT
from ..io.simpa.read_simpa import SimpaImporter

from ..utils import sort_key


def convert_ithera_msot_binary_to_hdf5(input_path: str, output_path: str, update: bool = False,
                                       use_user_defined_scan_name: bool = False):
    """
    This method converts all iThera MSOT files in a folder and exports them to an HDF5 format.

    :param use_user_defined_scan_name: Use a user defined scan name instead of the ithera number.
    :param input_path: A string that represents a directory. In this directory, a bunch of data is located in individual
                        folders following this naming convention: "Scan_N/" with N being an incrementing unique integer.
    :param output_path: The output path is a string representing a directory path pointing to the location where the
                        resulting hdf5 files should be stored.
    :param update: if True, the file will be processed again, even if an hdf5 file containing raw data with the desired
                   output name already exists at the output_path location.
    """
    scan_paths = list(sorted(glob.glob(join(input_path, "**", "Scan_*/"),
                                       recursive=True), key=sort_key))
    if len(scan_paths) < 1:
        print("Found no scans in the input path.")
    errors = []
    for scan_path in scan_paths:
        ithera_defined_scan_name = split(dirname(scan_path))[-1]
        print("WORKING ON", ithera_defined_scan_name)
        try:
            scan = iTheraMSOT(dirname(scan_path))
            if 0 in scan.raw_data.shape:
                print("Scan skipped because scan was not acquired properly", ithera_defined_scan_name)
        except FileNotFoundError:
            print("Scan skipped, because iThera scan corrputed.", ithera_defined_scan_name)
            errors.append("Scan skipped, because iThera scan corrputed. " + ithera_defined_scan_name)
            continue
        user_defined_scan_name = scan.get_scan_name()

        if use_user_defined_scan_name:
            scan_name = user_defined_scan_name
        else:
            scan_name = ithera_defined_scan_name

        scan_name = join(dirname(dirname(scan_path))[len(input_path) + 1:], scan_name)
        if os.path.exists(join(output_path, scan_name + ".hdf5")) and not update:
            continue
        print("SAVING AS", join(output_path, scan_name + ".hdf5"))

        scan.save_to_hdf5(join(output_path, scan_name + ".hdf5"))
        print("Saved to HDF5.")
    print("\n".join(errors))


def convert_simpa(input_path: str, output_path: str, update: bool = False,
                  use_user_defined_scan_name: bool = False, slice_n=None):
    """
    This method converts simpa files to HDF5 format.

    Parameters
    ----------
    input_path
    output_path
    update
    use_user_defined_scan_name
    slice_n
    """
    scan_paths = list(sorted(glob.glob(join(input_path, "**", "*.hdf5"),
                                       recursive=True), key=sort_key))
    if len(scan_paths) < 1:
        print("Found no scans in the input path.")

    i = 1
    for scan_path in scan_paths:
        print("WORKING ON", scan_path)

        scan = SimpaImporter(scan_path, z_slices=slice_n)
        user_defined_scan_name = scan.get_scan_name()

        if use_user_defined_scan_name:
            scan_name = user_defined_scan_name
        else:
            scan_name = f"Scan_{str(i)}"
            i += 1

        scan_name = join(dirname(dirname(scan_path))[len(input_path) + 1:], scan_name)
        print(scan_name)
        print("SAVING AS", join(output_path, scan_name + ".hdf5"))

        scan.save_to_hdf5(join(output_path, scan_name + ".hdf5"), update)
        print("Saved to HDF5.")
