#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse
import glob
from os.path import join, split

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RangeSlider
from .. import PAData
from .. import sort_key
from ..core.image_structures.single_parameter_data import SingleParameterData


def init_argparse():
    parser = argparse.ArgumentParser(description="View MSOT Recons.")
    parser.add_argument('input', type=str, help="Data Folder")
    parser.add_argument('-f', '--filter', type=str, help="Choose scan", default=None)
    parser.add_argument('-r', '--recon', type=int, help="Reconstruction number", default=0)
    parser.add_argument('-fn', '--filtername', type=str, help="Choose scan name filter", default=None)
    parser.add_argument('-t', '--thb', type=bool, help="Draw on THb", default=False)
    parser.add_argument('-s', '--so2', type=bool, help="Draw on SigSo2", default=False)
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    DATA_FOLDER = args.input
    for file in sorted(glob.glob(join(DATA_FOLDER, "*.hdf5")), key=sort_key):
        if args.filter is not None:
            if split(file)[-1] != "Scan_" + args.filter + ".hdf5":
                continue

        data = PAData.from_hdf5(file)

        if args.filtername is not None:
            if str.lower(args.filtername) not in str.lower(data.get_scan_name()):
                continue

        print(file)
        scan_name = data.get_scan_name()

        # by default just choose the first reconstruction.
        if not args.thb and not args.so2:
            reconstructions = data.get_scan_reconstructions()
        else:
            group_name = "thb" if args.thb else "so2"
            reconstructions = data.get_scan_images(group_name, SingleParameterData)

        if reconstructions == {}:
            print(f"{file} has not been reconstructed. Skipping.")
            continue

        methods = list(reconstructions.keys())

        if len(methods) > 1:
            print("Multiple reconstructions available, using", methods[0])

        recon_data = reconstructions[methods[0]]

        extents = recon_data.extent
        recon = recon_data.raw_data

        frame_n = 0
        wl = 0

        iqr = np.nanpercentile(recon[frame_n, wl], 95) - np.nanpercentile(recon[frame_n, wl], 5)
        median = np.median(recon[frame_n, wl])
        range_interest = (median - 3 * iqr, median + 3 * iqr)

        fig = plt.figure()
        plt.subplots_adjust(bottom=0.3)
        ax, ax2 = fig.subplots(1, 2)
        p = ax.imshow(np.squeeze(recon[frame_n, wl]), extent=extents, clim=range_interest)

        ax2.hist(recon[frame_n, wl].flatten())
        vlinea = ax2.axvline(np.nanmin(recon[frame_n, wl]))
        vlineb = ax2.axvline(np.nanmax(recon[frame_n, wl]))
        ax_clims = plt.axes([0.25, 0.15, 0.5, 0.03])
        ax_slide = plt.axes([0.25, 0.11, 0.5, 0.03])
        ax_frame = plt.axes([0.25, 0.07, 0.5, 0.03])
        wavelength = Slider(ax_slide, "Wavelength", 0, recon.shape[1] - 1, valinit=0, valstep=1)
        frame = Slider(ax_frame, "Frame", 0, recon.shape[0] - 1, valinit=0, valstep=1)
        clims = RangeSlider(ax_clims, "Clim Range", range_interest[0],
                            range_interest[1], valinit=range_interest)
        ax_text = plt.axes([0.3, 0.95, 0.4, 0.04])
        ax_text.axis("off")
        ax_text.annotate(scan_name, (0.5, 0), xycoords="axes fraction",
                         annotation_clip=False, horizontalalignment="center", verticalalignment="bottom")

        def update(_):
            nonlocal frame_n, wl
            frame_n = frame.val
            wl = wavelength.val
            p.set_data(np.squeeze(recon[frame_n, wl]))
            fig.canvas.draw()

        def update_clim(val):
            cmin = val[0]
            cmax = val[1]
            vlinea.set_data(([cmin, cmin], [0, 1]))
            vlineb.set_data(([cmax, cmax], [0, 1]))
            p.set_clim([cmin, cmax])

        frame.on_changed(update)
        wavelength.on_changed(update)
        clims.on_changed(update_clim)
        plt.show()
