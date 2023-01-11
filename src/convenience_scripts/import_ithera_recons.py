#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np


def init_argparse():
    parser = argparse.ArgumentParser(description="Import iThera recons.")
    parser.add_argument('input', type=str, help="Data File")
    parser.add_argument('binary', type=str, help="iThera Binary")
    parser.add_argument("description", type=str, help="Recon Description")
    parser.add_argument("fov", type=float, help="Field of View (metres) -e.g.0.025")
    parser.add_argument("-t", "--transpose", type=bool, help="Transpose Images", default=False)
    parser.add_argument("-r", "--reverse", type=int, nargs="*", help="Reverse Axes", default=[])
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    file = h5py.File(args.input, "r+")

    data = np.fromfile(args.binary, np.double)
    shape = file["raw_data"].shape[:-2]
    nx = int(np.sqrt(data.shape[0] // np.product(shape)))
    data = data.reshape(file["raw_data"].shape[:-2] + (nx, nx))

    if args.transpose:
        data = np.swapaxes(data, -1, -2)
    for r in args.reverse:
        if r == 0:
            data = data[:, :, ::-1, :]
        elif r == 1:
            data = data[:, :, :, ::-1]

    plt.imshow(data[0, 0])
    plt.show()

    if input("Is this the correct orientation?") == "Y":
        recs = file.require_group("recons")
        recs_g = recs.require_group(args.description)
        if "0" in recs_g:
            del recs_g["0"]
        rec = recs_g.create_dataset("0", data=data)
        rec.attrs["RECON_FOV"] = args.fov
    file.close()
