#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse
import glob
from os.path import join

import h5py
from .. import sort_key
from ..utils.mask_operations import generate_mask


def init_argparse():
    parser = argparse.ArgumentParser(description="Generate masks from rois for all data files in folder.")
    parser.add_argument('input', type=str, help="Data Folder")
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1"
    )
    parser.add_argument('-f', '--filter', type=int, help="Scan number choice.", default=None)
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    for filename in sorted(glob.glob(join(args.input, "*.hdf5")), key=sort_key):
        if int(filename.split(".")[0].split("_")[-1]) != args.filter and args.filter is not None:
            continue
        file = h5py.File(filename, "r+")
        if "rois" not in file:
            file.close()
            continue
        print(filename)
        # Generate mask and copy over all the attributes.
        # Do all the recon masks, then do all the unmixed masks.
        if "unmixed_masks" in file:
            del file["unmixed_masks"]
        if "recon_masks" in file:
            del file["recon_masks"]
        masks = file.require_group("recon_masks")
        for recon_type in file["recons"]:
            print(recon_type)
            maskgroup = masks.require_group(recon_type)
            for recon_num in file["recons"][recon_type]:
                masknumgroup = maskgroup.require_group(recon_num)
                dataset = file["recons"][recon_type][recon_num]
                # loop through all the rois:
                for roi in file["rois"]:
                    roigroup = masknumgroup.require_group(roi)
                    for roi_n in file["rois"][roi]:
                        verts = file["rois"][roi][roi_n]
                        if "RECON_FOV" not in dataset.attrs:
                            print("Warning recon has fov missing", recon_type, recon_num)
                        mask = generate_mask(verts[:], dataset.attrs.get("RECON_FOV", 0.025), dataset.shape[-1])
                        if roi_n in roigroup:
                            del roigroup[roi_n]
                        roidata = roigroup.create_dataset(roi_n, data=mask)
                        for att in verts.attrs:
                            roidata.attrs[att] = verts.attrs[att]
        if "unmixed" in file:
            masks = file.require_group("unmixed_masks")
            for recon_type in file["unmixed"]:
                maskgroup = masks.require_group(recon_type)
                for recon_num in file["unmixed"][recon_type]:
                    masknumgroup = maskgroup.require_group(recon_num)
                    dataset = file["unmixed"][recon_type][recon_num]
                    # loop through all the rois:
                    for roi in file["rois"]:
                        roigroup = masknumgroup.require_group(roi)
                        for roi_n in file["rois"][roi]:
                            verts = file["rois"][roi][roi_n]
                            if roi_n in roigroup:
                                del roigroup[roi_n]
                            if "RECON_FOV" not in dataset.attrs:
                                print("Warning unmixed has fov missing", recon_type, recon_num)
                            mask = generate_mask(verts[:], dataset.attrs.get("RECON_FOV", 0.025), dataset.shape[-1])
                            roidata = roigroup.create_dataset(roi_n, data=mask)
                            for att in verts.attrs:
                                roidata.attrs[att] = verts.attrs[att]
        file.close()
