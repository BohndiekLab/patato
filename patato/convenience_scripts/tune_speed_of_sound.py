#  Copyright (c) Thomas Else 2023.
#  License: BSD-3
"""
``msot-set-speed-of-sound``: This module contains functions for setting the speed of sound for all datasets in a
directory.
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RangeSlider
from .. import PAData

try:
    from pyopencl.array import Array
except ImportError:
    Array = None

from .. import read_reconstruction_preset
from .. import MSOTPreProcessor
from .. import ReconstructionAlgorithm, get_default_recon_preset, OpenCLBackprojection
from .. import sort_key


def main():
    parser = argparse.ArgumentParser(description="Process MSOT Data.")
    parser.add_argument('input', type=str, help="Data Folder")
    parser.add_argument('c', type=float, help="Speed of Sound")
    parser.add_argument('-p', '--preset', type=str, help="Preset File",
                        default=None)
    parser.add_argument('-r', '--run', type=int, help="Run Number",
                        default=None)
    parser.add_argument('-c', '--clear', type=bool, help="Clear saved speeds",
                        default=False)
    parser.add_argument('--console', type=bool, help="Show interactive interface?",
                        default=False)
    parser.add_argument('-s', '--scan', type=str, help="Scan Number",
                        default=None)
    parser.add_argument('-b', '--startscan', type=int, help="First Scan Number",
                        default=None)
    parser.add_argument('-w', '--wavelength', type=float, help="Preview Wavelength (nm)",
                        default=None)
    parser.add_argument('-u', '--usthreshold', type=float, help="Overlay Recon on Ultrasound",
                        default=None)
    parser.add_argument('-l', '--log', type=bool, help="Log Scaling")
    parser.add_argument('-L', '--lineplots', type=bool, help="Show Line Plots")
    args = parser.parse_args()

    preset_arg = args.preset

    wavelength = args.wavelength
    if wavelength is None:
        wavelength = 800.

    data_folder = args.input
    speed_of_sound = args.c

    scan_number = args.scan
    clear = args.clear
    run_number = args.run
    if run_number is None:
        run_number = 0

    # TODO: implement ultrasound threshold overlay
    us_threshold = args.usthreshold
    log = args.log

    files = sorted(glob.glob(os.path.join(data_folder, "**", "*.hdf5"), recursive=True), key=sort_key)

    for data_file in files:
        if args.startscan is not None:
            if int(data_file.split("_")[-1].split(".")[0]) < args.startscan:
                print("Skipping", data_file)
                continue
        if str(scan_number) not in data_file and scan_number is not None:
            continue

        print(data_file)
        pa_data = PAData.from_hdf5(data_file, "r+")[:10]
        if any(x == 0 for x in pa_data.dataset.shape):
            continue
        preset = preset_arg
        if preset_arg is None:
            preset = get_default_recon_preset(pa_data)

        with open(preset) as json_file:
            settings = json.load(json_file)

        pipeline = read_reconstruction_preset(settings)

        preprocessor: MSOTPreProcessor = pipeline
        preprocessor.time_factor = 1
        preprocessor.detector_factor = 1
        reconstructor: ReconstructionAlgorithm = pipeline.children[0]

        if pa_data.get_speed_of_sound() is not None and not clear:
            continue
        if pa_data.get_speed_of_sound() is not None:
            speed_of_sound = pa_data.get_speed_of_sound()

        if args.console:
            print(f"Setting speed of sound for {data_file}")
            pa_data.set_speed_of_sound(speed_of_sound)
            continue

        wl_index = np.argmin(np.abs(wavelength - pa_data.get_wavelengths()))
        pa_data = pa_data[:, wl_index:wl_index + 1]
        pre_processed, recon_args, _ = preprocessor.run(pa_data.get_time_series(),
                                                        pa_data)

        # print(pre_processed.raw_data.shape)
        # plt.imshow(np.squeeze(pre_processed.raw_data[0]))
        # plt.show()
        # E.g. transfer to GPU if necessary
        # pre_processed = reconstructor.pre_prepare_data(pre_processed)

        recon_info, _, _ = reconstructor.run(pre_processed[run_number:run_number + 1],
                                             pa_data[run_number:run_number + 1],
                                             speed_of_sound=speed_of_sound, **recon_args)
        recon = recon_info.raw_data

        # From here down, it's mostly Matplotlib Code.
        oa_cmap = "magma"

        extent = recon_info.extent

        fig = plt.figure(figsize=(12, 6))

        plt.suptitle(f"Tuning speed of sound for {data_file}")
        plt.subplots_adjust(bottom=0.3)
        n_plots = 3 if args.lineplots else 2
        axes = fig.subplots(1, n_plots)
        ax = axes[0]
        ax2 = axes[1]
        ax3 = None
        if n_plots == 3:
            ax3 = axes[2]

        ax_slice = plt.axes([0.25, 0.2, 0.65, 0.03])
        ax_clims = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_button = plt.axes([0.25, 0.025, 0.4, 0.05])

        image = np.squeeze(recon)
        if log:
            image = np.log(image)

        image_ranges = image[np.isfinite(image) & ~np.isnan(image)]

        ax2.hist(image_ranges)
        v_line_a = ax2.axvline(np.nanmin(image))
        v_line_b = ax2.axvline(np.nanmax(image))

        p = ax.imshow(image, cmap=oa_cmap, extent=extent, origin="lower")
        xlabels = np.linspace(extent[0], extent[1], image.shape[0])
        ylabels = np.linspace(extent[2], extent[3], image.shape[1])
        if ax3 is not None:
            ax3.plot(xlabels, image[int(image.shape[0] / 2), :])
            ax3.plot(ylabels, image[:, int(image.shape[1] / 2)])

        speed = Slider(ax_slide, "Speed of Sound", 1400, 1600, valinit=speed_of_sound, valstep=1)
        image_slice = Slider(ax_slice, "Frame Number", 0, pa_data.shape[0] - 1, valinit=run_number, valstep=1)
        save_btn = Button(ax_button, "Save Speed of Sound")
        clims = RangeSlider(ax_clims, "Clim Range", np.nanmin(image_ranges), np.nanmax(image_ranges))

        def update(_):
            nonlocal pre_processed, recon_args, speed_of_sound, speed, recon, reconstructor, p, recon, log, ax3, image, pa_data
            c_new = speed.val
            recon_info, _, _ = reconstructor.run(pre_processed[image_slice.val:image_slice.val + 1],
                                                       pa_data[image_slice.val:image_slice.val + 1],
                                                       speed_of_sound=c_new,
                                                       **recon_args)
            recon = recon_info.raw_data
            image = np.squeeze(recon)
            if log:
                image = np.log(image)
            p.set_data(image)
            if ax3 is not None:
                ax3.clear()
                ax3.plot(xlabels, image[int(image.shape[0] / 2), :])
                ax3.plot(ylabels, image[:, int(image.shape[1] / 2)])
            fig.canvas.draw()

        def save(_):
            nonlocal pa_data, speed
            c_set = speed.val
            pa_data.set_speed_of_sound(c_set)
            print("Saved speed of sound")

        def update_clim(val):
            nonlocal p, v_line_a, v_line_b
            cmin = val[0]
            cmax = val[1]
            v_line_a.set_data(([cmin, cmin], [0, 1]))
            v_line_b.set_data(([cmax, cmax], [0, 1]))
            p.set_clim([cmin, cmax])

        save_btn.on_clicked(save)
        speed.on_changed(update)
        clims.on_changed(update_clim)
        image_slice.on_changed(update)

        def on_close(_):
            nonlocal pre_processed, reconstructor
            if type(pre_processed.raw_data) == Array:
                pre_processed.raw_data.data.release()
            if type(reconstructor) == OpenCLBackprojection:
                reconstructor.queue.finish()

        fig.canvas.mpl_connect('close_event', on_close)
        plt.show()


if __name__ == "__main__":
    main()
