#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse
import glob
import re
from os.path import join

import h5py
import numpy as np
from .. import sort_key


def init_argparse():
    map = {"TRUE": True, "FALSE": False, "T": True, "F": False, "YES": True, "NO": False, "Y": True, "N": False}
    map_fn = lambda x: map[x.upper()]
    parser = argparse.ArgumentParser(description="Copy ROIs between scans made of same thing at same time.")
    parser.add_argument('input', type=str, help="Data Folder")
    parser.add_argument('-r', '--regex', default=None, type=str, help="Regex for parsing name")
    parser.add_argument('-c', '--copyclose', default=False, type=map_fn, help="Automatically copy over roi to closest "
                                                                              "slice if not exact matches (tolerance "
                                                                              "1mm)")
    parser.add_argument('-d', '--deleteold', default=True, type=map_fn, help="Delete old copies.")
    parser.add_argument('-j', '--justdeletecopies', default=False, type=map_fn,
                        help="Just delete copies of rois.")
    parser.add_argument('-f', '--copyfrom', default=None,
                        help="Copy only from scan type (e.g. OE).")
    parser.add_argument('-t', '--copyto', default=None,
                        help="Copy only to scan type (e.g. DCE).")
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    DATA_FOLDER = args.input

    regex = args.regex

    if args.regex is None:
        regex = r"^(?P<mouse>[^_\- ]+)[_\- ](?P<ear>[^_\- ]+)[_\- ]((?P<scan_number>[0-9]+)[_\- ])?(?P<scan_type>.+)$"
    elif args.regex == "tom":
        regex = r"^(?P<date>[0-9]+)\-(?P<mouse>.+)\-(?P<scan_type>.+)"
    elif args.regex == "marilena":
        regex = r"VMstudy_MEO#(?P<number>[0-9]+)_((Day(?P<day>[0-9]+))|MSOT)_?(?P<scan_type>GS|SS|MS|GC|again)?"

    regex = re.compile(regex)

    scan_groups = {}

    for file in sorted(glob.glob(join(DATA_FOLDER, "**", "*.hdf5"), recursive=True), key=sort_key):
        data = h5py.File(file, "r")
        name_regex = regex.fullmatch(data["raw_data"].attrs["name"])
        print(file, data["raw_data"].attrs["name"], name_regex)
        if name_regex is None:
            continue
        name_regex = name_regex.groupdict()
        scan_group = []
        for k in name_regex:
            if k != "scan_type":
                if name_regex[k] is not None:
                    scan_group.append(name_regex[k])
                else:
                    scan_group.append("1")
        scan_group = "-".join(scan_group)
        if scan_group not in scan_groups:
            scan_groups[scan_group] = [file]
        else:
            scan_groups[scan_group].append(file)
        data.close()

    for group in scan_groups:
        rois = []
        roi_attrs = []
        roi_names = []
        roi_numbers = []
        roi_scan_origin = []
        for file in scan_groups[group]:
            data = h5py.File(file, "r+")
            if args.copyfrom is not None:
                if args.copyfrom not in data["raw_data"].attrs["name"]:
                    continue
            if "rois" in data:
                for a in data["rois"]:
                    deletion = False  # track whether any rois deleted so we can renumber at the end
                    for b in data["rois"][a]:
                        if data["rois"][a][b].attrs.get("copy", default=False):
                            if args.deleteold or args.justdeletecopies:
                                del data["rois"][a][b]
                                deletion = True
                            continue
                        rois.append(data["rois"][a][b][:])
                        roi_attrs.append(dict(data["rois"][a][b].attrs))
                        roi_names.append(a)
                        roi_numbers.append(b)
                        roi_scan_origin.append(data["raw_data"].attrs["name"])
                    if deletion:
                        original = sorted(list(data["rois"][a].keys()), key=int)
                        change = [str(x) for x in range(len(original))]
                        for old, new in zip(original, change):
                            data["rois"][a].move(old, new)
            data.close()
        for file in scan_groups[group]:
            data = h5py.File(file, "r+")
            if args.copyto is not None:
                if args.copyto not in data["raw_data"].attrs["name"]:
                    continue
            for roi, att, nam, num, ori in zip(rois, roi_attrs, roi_names, roi_numbers, roi_scan_origin):
                if ori != data["raw_data"].attrs["name"]:
                    zs = np.unique(data["Z-POS"][:])
                    make_copy = False
                    if att["z"] in zs:
                        # print("Able to copy from", ori, data["raw_data"].attrs["name"])
                        make_copy = True
                    elif np.any(np.isclose(zs, att["z"], atol=1, rtol=0)):
                        print("Not exact, but close for: ", ori, "to", data["raw_data"].attrs["name"])
                        copy = input("Not exact match, copy anyway? Y/[N]") if not args.copyclose else True
                        if copy in ["Y", "y", True]:
                            make_copy = True
                            old_z = att["z"]
                            att["z"] = zs[np.argmin(np.abs(zs - att["z"]))]
                            print("Copying, changing z from {:.2f} to {:.2f}".format(old_z, att["z"]))
                    else:
                        # print("No close positions for: ", ori, "to", data["raw_data"].attrs["name"], "so skipping.")
                        # print(zs[np.argmin(np.abs(zs - att["z"]))], att["z"])
                        # print(np.isclose(zs, att["z"], atol=0.1, rtol=0))
                        make_copy = False
                    if make_copy and not args.justdeletecopies:
                        # copy the roi to scan
                        roi_grp = data.require_group("rois")
                        roi_grp = roi_grp.require_group(nam)
                        roi_num = str(len(roi_grp.keys()))
                        roi_data = roi_grp.create_dataset(roi_num, data=roi)
                        roi_data.attrs["copy"] = True
                        for a in att:
                            roi_data.attrs[a] = att[a]
            data.close()
