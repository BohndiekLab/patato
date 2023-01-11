#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

# MAKE Measurements from ROIS :DDDDD
import argparse
import glob
from os.path import join

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ...core.image_structures import image_structure_types
from ...utils import sort_key
from ...utils.mask_operations import generate_mask
from scipy.signal import fftconvolve


def find_dce_boundaries(scan, reference_name="reference_",
                        recon="Backprojection Preclinical", recon_n="0ICG", window=10, display=False, sigma=2):
    if recon is None:
        recon = list(sorted(scan["recons"].keys()))[0]
    data_icg = scan["unmixed"][recon][recon_n]

    fov = data_icg.attrs["RECON_FOV"]
    masks = []
    if "rois" in scan:
        for ref_number in sorted(scan["rois"][reference_name].keys()):
            reference_roi = scan["rois"][reference_name][ref_number]
            mask = generate_mask(reference_roi[:], fov, data_icg.shape[-1])
            masks.append(mask)
    elif "unmixed_masks" in scan:
        for ref_number in sorted(scan["unmixed_masks"][recon]["0"][reference_name].keys()):
            mask = scan["unmixed_masks"][recon]["0"][reference_name][ref_number][:]
            masks.append(mask)
    mask = masks[-1]
    measurement = image_structure_types.T

    kernel = np.arange(window) - window / 2 + 1 / 2
    kernel = np.exp(-(kernel / sigma) ** 2)
    kernel /= np.sum(kernel)
    smoothed = fftconvolve(measurement, kernel[:, None], "valid")
    smoothed_grad = np.median(np.gradient(smoothed, axis=0), axis=-1)
    steps = [0]
    # Find first peak.
    step_point = np.argmax(smoothed_grad)
    steps.append(step_point + window // 2)
    steps.append(len(measurement) - 1)
    if display:
        plt.plot(np.median(smoothed, axis=-1))
        plt.twinx()
        plt.plot(smoothed_grad, c="C1")
        for s in steps:
            plt.axvline(s - window // 2)
        plt.show()
    return steps



def init_argparse():
    parser = argparse.ArgumentParser(description="Analyse Gas Challenge Data.")
    parser.add_argument('input', type=str, help="Data Folder")
    parser.add_argument('-p', '--prefix', type=str, help="Gas Challenge name prefix")
    parser.add_argument('-w', '--window', type=int, help="Smoothing window size",
                        default=10)
    parser.add_argument('-b', '--buffer', type=int, help="Buffer around changes",
                        default=5)
    parser.add_argument('-d', '--display', type=bool, help="Display steps",
                        default=False)
    parser.add_argument('-s', '--sigma', type=float, help="Smoothing window sigma",
                        default=4)
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    DATA_FOLDER = args.input
    prefix = args.prefix
    window = args.window
    buffer = args.buffer

    for file in sorted(glob.glob(join(DATA_FOLDER, "*.hdf5")), key=sort_key):
        data = h5py.File(file, "r+")
        scan_name = data["raw_data"].attrs["name"]
        if prefix.lower() not in scan_name.lower():
            continue
        if "rois" not in data and "unmixed_masks" not in data:
            print("Skipped", file, "- no rois.")
            continue
        if "rois" in data:
            if "reference_" not in data["rois"]:
                print("Skipped", file, "- no reference.")
                continue
        elif "unmixed_masks" in data:
            if "reference_" not in data["unmixed_masks/Backprojection Preclinical/0"]:
                print("Skipped", file, "- no reference.")
                continue
        else:
            continue
        if "unmixed" not in data:
            print("Skipped", file, "- no unmixed data.")
            continue
        print(file)
        steps = find_dce_boundaries(data, window=window, display=args.display, sigma=args.sigma)
        if args.display:
            if input("Continue with analysis?") != "Y":
                continue
        # Otherwise process dce data.
        icg_grp = data.require_group("baseline_icg")
        sicg_grp = data.require_group("baseline_icg_sigma")
        dicg_grp = data.require_group("dicg")
        dicg_grp.attrs["steps"] = steps
        dicg_grp.attrs["buffer"] = buffer
        for method in data["so2"]:
            icg_mgrp = icg_grp.require_group(method)
            sicg_mgrp = sicg_grp.require_group(method)

            dicg_mgrp = dicg_grp.require_group(method)
            for recon in data["unmixed"][method]:
                if "ICG" not in recon:
                    continue
                data_icg = data["unmixed"][method][recon]
                if data_icg.shape[0] == 1:
                    continue
                baseline = np.mean(data_icg[steps[0]:steps[1] - buffer, 2], axis=0)
                delta_icg = np.max(data_icg[steps[1] + buffer:steps[2] - buffer, 2],
                                   axis=0) - baseline
                sigma_baseline = np.std(data_icg[steps[0]:steps[1] - buffer, 2], axis=0)
                if recon in dicg_mgrp:
                    del dicg_mgrp[recon]
                if recon in icg_mgrp:
                    del icg_mgrp[recon]
                if recon in sicg_mgrp:
                    del sicg_mgrp[recon]
                dicg_set = dicg_mgrp.create_dataset(recon, data=delta_icg)
                icg_set = icg_mgrp.create_dataset(recon, data=baseline)
                sicg_set = sicg_mgrp.create_dataset(recon, data=sigma_baseline)
                for a in data_icg.attrs:
                    dicg_set.attrs[a] = data_icg.attrs[a]
        data.close()
