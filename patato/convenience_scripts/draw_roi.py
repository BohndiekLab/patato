#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

"""
``msot-draw-roi``: A script to draw regions of interest interactively on datasets..
"""

import argparse
import glob
import sys
from collections import namedtuple
from os.path import join, split

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RangeSlider, PolygonSelector
from .. import ImageSequence
from .. import ROI, PAData
from .. import sort_key
from scipy.optimize import least_squares

if matplotlib.get_backend() == "MacOSX":
    matplotlib.use("TkAgg")

from ..utils.roi_operations import ROI_NAMES, REGION_COLOURS, close_loop

# TODO: Temporary fix for ?
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def circ_loss(x, points):
    R = x[0]
    x_0 = x[1]
    y_0 = x[2]
    return np.sum(((points[:, 0] - x_0) ** 2 / R ** 2 + (points[:, 1] - y_0) ** 2 / R ** 2 - 1) ** 2)


def get_circle_points(r, x_0, y_0, n_points=50):
    thetas = np.linspace(0, 2 * np.pi, n_points, False)
    return np.array([np.sin(thetas) * r + x_0, np.cos(thetas) * r + y_0]).T


def fitcirc(points):
    points = np.array(points)
    x_0 = np.mean(points[:, 0])
    y_0 = np.mean(points[:, 1])
    r_0 = np.sqrt(np.mean((points[:, 0] - x_0) ** 2 + (points[:, 1] - y_0) ** 2))
    fit = least_squares(circ_loss, x0=np.array([r_0, x_0, y_0]), args=(points,))
    return get_circle_points(*fit.x)


positions = ["left", "right", "full", "", "radial", "ulnar", "superficial"]


def init_argparse():
    parser = argparse.ArgumentParser(description="Process MSOT Data.")
    parser.add_argument('input', type=str, help="Data Folder")
    parser.add_argument('-n', '--name', type=str, help="ROI Name",
                        choices=ROI_NAMES, default="unnamed")
    parser.add_argument('-p', '--position', type=str, help="ROI Location",
                        choices=positions, default="")
    parser.add_argument('-f', '--filter', type=str, help="Choose scan", default=None)
    parser.add_argument('-r', '--recon', type=int, help="Reconstruction number", default=0)
    parser.add_argument('-c', '--clear', type=bool, help="Clear Rois", default=False)
    parser.add_argument('-fn', '--filtername', type=str, help="Choose scan name filter", default=None)
    parser.add_argument('-thb', '--drawthb', type=bool, help="Draw on THb", default=False)
    parser.add_argument('-s', '--drawsigso2', type=bool, help="Draw on SigSo2", default=False)
    parser.add_argument('-u', '--drawultrasound', type=bool, help="Draw on Ultrasound", default=False)
    parser.add_argument('-o', '--overlaythb', type=bool, help="Draw on THb", default=False)
    parser.add_argument('-t', '--overlaythresh', type=float, help="THb threshold", default=80.)
    parser.add_argument('-fc', '--fitcircles', type=bool, help="Fit Circles To Rois", default=False)
    parser.add_argument('-l', '--logoverlay', type=bool, help="Log of overlay", default=False)
    parser.add_argument('-g', '--start_scan_number', type=int, help="Start drawing at scan n", default=None)
    parser.add_argument('-i', '--interpolation', type=bool, help="Interpolate between ROI slices", default=False)
    return parser


class ROIDrawer:
    def update(self, _=None):
        if self.previous_frame_number != self.i_index:
            self.selection_n = -1
        if self.selection_n == -1:
            self.edit_btn.label.set_text("No ROI Selected")
            self.edit_btn.set_active(False)
            self.delete_btn.label.set_text("No ROI Selected")
            self.delete_btn.set_active(False)
        else:
            self.edit_btn.label.set_text("Change ROI Name")
            self.edit_btn.set_active(True)
            self.delete_btn.label.set_text("Delete ROI")
            self.delete_btn.set_active(True)
        self.previous_frame_number = self.i_index

        # Update the images that are shown
        self._draw_image()
        # self.main_image.set_data(np.squeeze(self.image_data[self.i_index, self.j_index].raw_data))
        # if self.overlay_data is not None:
        #     self.overlay_image.set_data(np.squeeze(self.overlay_data[self.i_index, self.j_index].raw_data))

        # remove the roi lines
        for r in self.roi_lines:
            line = r.pop(0)
            line.remove()

        self.roi_lines = []
        self.roi_names = []

        # Plot the regions.
        roi_number = 0
        for (r, n), roi_data in self.pa_data.get_rois(interpolate=self.interpolation).items():
            colour = "C0"
            for k, c in zip(ROI_NAMES, REGION_COLOURS):
                if k in r:
                    colour = c

            roi_points = roi_data.points

            if np.isclose(self.match_frames[self.i_index, self.j_index],
                          roi_data.attributes.get(self.frame_type, 1.0)):
                roi_close = close_loop(roi_points)
                roi_plot = self.image_axis.plot(roi_close[:, 0], roi_close[:, 1], picker=True,
                                                label=r + "/" + n, c=colour, scalex=False,
                                                scaley=False)
                # Show the ROI on the appropriate frame.
                if roi_number == self.selection_n:
                    roi_plot[0].set_linestyle("dashed")
                    roi_plot[0].set_zorder(100)
                self.roi_lines.append(roi_plot)
                self.roi_names.append(r + "/" + n)
                roi_number += 1

        frame_number = 0
        # Draw the ROI names
        for lin, nam in zip(self.roi_lines, self.roi_names):
            if frame_number >= len(self.legend):
                self.legend.append(
                    self.fig.text(self.legend_x_0, self.legend_y_0 + self.legend_y,
                                  nam, color=lin[0].get_color(), picker=True))
                self.legend_y += self.legend_dy
            else:
                self.legend[frame_number].set_text(nam)
                self.legend[frame_number].set_color(lin[0].get_color())
            if frame_number == self.selection_n:
                self.legend[frame_number].set_fontweight("extra bold")
            else:
                self.legend[frame_number].set_fontweight("normal")
            frame_number += 1

        while frame_number < len(self.legend):
            self.legend[frame_number].set_text("")
            frame_number += 1

        # Add some labels.
        z = self.pa_data.get_z_positions()[self.i_index, self.j_index]
        self.z_text.set_text(f"z = {z:.2f} mm")
        wavelength = self.pa_data.get_wavelengths()[self.j_index]
        self.wl_text.set_text(f"wavelength = {wavelength:.0f} nm")

        # Reset the colour limits
        a, b = self.ax_clims.get_xlim()
        self.ax_clims.set_xlim((min([np.nanmin(self.main_image.get_array()), a]),
                                max([np.nanmax(self.main_image.get_array()), b])))

        # Redraw
        if plt.get_backend() == "MacOSX":
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()

    def on_pick(self, event):
        if not self.drawing:
            line = event.artist
            try:
                name = line.get_text()
            except AttributeError:
                name = line.get_label()
            if self.selection_n != self.roi_names.index(name):
                self.selection_n = self.roi_names.index(name)
            else:
                self.selection_n = -1
            self.update()

    def _draw_histogram(self):
        self.histogram = self.hist_axis.hist(self.image_data[self.i_index, self.j_index].values.flatten(),
                                             bins=20, density=True)

    def _draw_hist_lines(self):
        self.vlinea = self.hist_axis.axvline(np.nanmin(self.image_data[self.i_index, self.j_index].raw_data))
        self.vlineb = self.hist_axis.axvline(np.nanmax(self.image_data[self.i_index, self.j_index].raw_data))

    def _draw_image(self):
        self.main_image = self.image_axis.imshow(np.squeeze(self.image_data[self.i_index, self.j_index].raw_data),
                                                 extent=self.extent, cmap=self.cmap, origin="lower")
        if self.overlay_data is not None:
            overlay_image = np.full_like(self.overlay_data[self.i_index, self.j_index].raw_data, np.nan)
            overlay_image[:] = self.overlay_data[self.i_index, self.j_index].raw_data
            overlay_image[overlay_image < self.overlay_threshold] = np.nan
            self.overlay_image = self.image_axis.imshow(np.squeeze(overlay_image),
                                                        extent=self.overlay_extent, cmap=self.overlay_cmap,
                                                        origin="lower")

    @property
    def i_index(self):
        try:
            return self.frame.val
        except AttributeError:
            return 0

    @property
    def j_index(self):
        try:
            return self.wavelength.val
        except AttributeError:
            return 0

    def save_roi(self, val):
        if self.drawing:
            z = self.pa_data.get_z_positions()[self.i_index, self.j_index]
            run = self.pa_data.get_run_number()[self.i_index, self.j_index]
            repetition = self.pa_data.get_repetition_numbers()[self.i_index, self.j_index]
            roi_name = self.roi_name
            roi_position = self.roi_position
            roi_output = self.verts
            self.pa_data.add_roi(ROI(roi_output, z, run, repetition, roi_name, roi_position))
            print("Saved ROI", self.roi_name, self.scan_name)
            self.p_selector.set_visible(False)
            self.drawing = False
            self.start_btn.label.set_text("Start ROI")
            self.update()
            self.p_selector = None

    def update_clim(self, val):
        cmin = val[0]
        cmax = val[1]
        self.vlinea.set_data(([cmin, cmin], [0, 1]))
        self.vlineb.set_data(([cmax, cmax], [0, 1]))
        self.main_image.set_clim([cmin, cmax])

    def poly_onselect(self, v):
        self.verts = v

    def start_drawing(self, v):
        if not self.drawing:
            self.selection_n = -1
            self.drawing = True
            self.p_selector = PolygonSelector(self.image_axis, self.poly_onselect)
            self.start_btn.label.set_text("Cancel ROI")
            self.update()
        else:
            self.selection_n = -1
            self.drawing = False
            self.p_selector.set_visible(False)
            self.p_selector = None
            self.start_btn.label.set_text("Start ROI")
            self.update()

    def edit_roi_name(self, _=None):
        if self.selection_n == -1:
            return
        name = self.roi_names[self.selection_n]
        self.pa_data.rename_roi(name, self.roi_name, self.roi_position)
        self.selection_n = -1
        self.update()

    def delete_roi(self, _=None):
        name = self.roi_names[self.selection_n]
        print("Deleting", name)
        self.pa_data.delete_rois(*name.split("/"))
        self.selection_n = -1
        self.update()

    def on_close(self, _=None):
        self.pa_data.close()

    def on_key_press(self, event):
        sys.stdout.flush()
        if event.key == "up":
            self.selection_n += 1
            self.selection_n %= len(self.roi_lines)
        elif event.key == "down":
            self.selection_n -= 1
            if self.selection_n < -1:
                self.selection_n = -1
        if event.key == "left":
            self.frame.set_val(max(self.frame.val - 1, self.frame.valmin))
        elif event.key == "right":
            self.frame.set_val(min(self.frame.val + 1, self.frame.valmax))
        if event.key == "r":
            self.edit_roi_name()
        self.update()

    def __init__(self, pa_data, image_data: ImageSequence, image_extent, overlay=None,
                 overlay_threshold=None, overlay_extent=None,
                 roi_name="unnamed", roi_position="", interpolation=False):
        clinical = pa_data.is_clinical()
        self.frame_type = "z" if not clinical else "repetition"
        self.match_frames = pa_data.get_z_positions() if not clinical else pa_data.get_repetition_numbers()
        self.pa_data = pa_data
        self.interpolation = interpolation

        self.image_data = image_data
        self.overlay_data = overlay
        self.overlay_threshold = overlay_threshold
        self.overlay_extent = overlay_extent or image_extent
        self.overlay_cmap = None
        if overlay is not None:
            self.overlay_cmap = overlay.cmap
        self.fig = plt.figure()
        self.image_axis = plt.subplot2grid((2, 2), (0, 0), 2, fig=self.fig)
        self.hist_axis = plt.subplot2grid((2, 2), (0, 1), fig=self.fig)
        self.hist_axis.set_ylabel("Fraction")
        self.hist_axis.set_xlabel("PA Intensity")
        plt.tight_layout(pad=3)
        plt.subplots_adjust(bottom=0.3)
        self.extent = image_extent
        self.cmap = image_data.cmap
        self.main_image = None
        self.overlay_image = None
        self.histogram = None

        # Draw the main images
        self._draw_image()

        self._draw_histogram()
        self.vlinea = None
        self.vlineb = None
        self._draw_hist_lines()
        # Define the slider and button axes
        self.ax_clims = plt.axes([0.25, 0.15, 0.5, 0.03])
        self.ax_slide = plt.axes([0.25, 0.11, 0.5, 0.03])
        self.ax_frame = plt.axes([0.25, 0.07, 0.5, 0.03])
        self.ax_button = plt.axes([0.25, 0.01, 0.2, 0.05])
        self.ax_start = plt.axes([0.55, 0.01, 0.2, 0.05])

        if self.image_data.shape[1] != 1:
            self.wavelength = Slider(self.ax_slide, "Wavelength", 0,
                                     self.image_data.shape[1] - 1, valinit=0, valstep=1)
        else:
            Constant = namedtuple("ConstantWidget", ["val", "on_changed"])
            self.wavelength = Constant(val=0, on_changed=lambda x: None)
            self.ax_slide.set_visible(False)

        self.frame = Slider(self.ax_frame, "Frame", 0, self.image_data.shape[0] - 1, valinit=0, valstep=1)
        self.previous_frame_number = 0

        self.roi_lines = []
        self.roi_names = []
        self.image_axis.set_prop_cycle(None)
        self.legend = []
        self.legend_y_0 = 0.55
        self.legend_dy = -0.04
        self.legend_x_0 = 0.55
        self.legend_y = 0
        self.selection_n = -1
        self.drawing = False

        self.ax_edit = plt.axes([self.legend_x_0 + 0.2, self.legend_y_0 - 0.1, 0.2, 0.05])
        self.edit_btn = Button(self.ax_edit, "No ROI Selected")
        self.edit_btn.set_active(False)

        self.ax_delete = plt.axes([self.legend_x_0 + 0.2, self.legend_y_0 - 0.3, 0.2, 0.05])
        self.delete_btn = Button(self.ax_delete, "No ROI Selected")
        self.delete_btn.set_active(False)

        self.save_btn = Button(self.ax_button, "Save ROI")
        self.start_btn = Button(self.ax_start, "Start ROI")

        self.clims = RangeSlider(self.ax_clims, "Clim Range",
                                 np.nanmin(self.image_data[self.i_index, self.j_index].raw_data),
                                 np.nanmax(self.image_data[self.i_index, self.j_index].raw_data))

        self.clims.set_min(np.nanmin(self.main_image.get_array()))
        self.clims.set_max(np.nanmax(self.main_image.get_array()))
        self.clims.set_val((np.nanmin(self.main_image.get_array()), np.nanmax(self.main_image.get_array())))

        self.ax_text = plt.axes([0.3, 0.95, 0.4, 0.04])
        self.ax_text.axis("off")
        self.scan_name = pa_data.get_scan_name()
        self.roi_name = roi_name
        self.roi_position = roi_position
        self.ax_text.annotate(self.scan_name + ": " + self.roi_name + " " + self.roi_position, (0.5, 0),
                              xycoords="axes fraction", annotation_clip=False,
                              horizontalalignment="center", verticalalignment="bottom")

        Constant = namedtuple("ConstantWidget", ["set_text"])
        self.z_text = Constant(lambda x: None)
        z = pa_data.get_z_positions()[self.i_index, self.j_index]
        if not np.isnan(z):
            self.z_text = self.fig.text(0.6, 0.19, f"z = {z:.2f} mm")

        wl = pa_data.get_wavelengths()[self.j_index]
        self.wl_text = self.fig.text(0.3, 0.19, f"wavelength = {wl:.0f} nm")

        self.p_selector = None
        self.verts = None
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.update()
        self.update_clim(self.main_image.get_clim())

        self.frame.on_changed(self.update)
        self.wavelength.on_changed(self.update)
        self.save_btn.on_clicked(self.save_roi)
        self.start_btn.on_clicked(self.start_drawing)
        self.clims.on_changed(self.update_clim)
        self.edit_btn.on_clicked(self.edit_roi_name)
        self.delete_btn.on_clicked(self.delete_roi)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        # fm = self.fig.canvas.manager.full_screen_toggle()
        plt.show()


def main():
    p = init_argparse()
    args = p.parse_args()

    DATA_FOLDER = args.input
    print(args.input)
    for file in sorted(glob.glob(join(DATA_FOLDER, "**", "*.hdf5"), recursive=True), key=sort_key):
        if args.filter is not None:
            if split(file)[-1] != "Scan_" + args.filter + ".hdf5":
                continue
        if args.start_scan_number is not None:
            file_name = split(file)[-1]
            if "_" in file_name:
                number = int(file_name.split("_")[1].split(".")[0])
                if number < args.start_scan_number:
                    continue

        data = PAData.from_hdf5(file, "r+")

        if args.filtername is not None:
            if str.lower(args.filtername) not in str.lower(data.get_scan_name()):
                continue
            print(file)

        if args.clear:
            if args.name == "unnamed" and args.position == "":
                data.delete_rois()
            else:
                data.delete_rois(args.name + "_" + args.position)
            continue

        if not data.get_recon_types():
            print(f"Skipping {file} because no reconstructions are present.")
            continue

        print(split(file)[-1])

        methods = data.get_recon_types()
        print(methods)
        method = sorted(methods)[args.recon]

        if len(methods) > 1:
            print(f"Multiple reconstructions available, using {method}.")

        recon = data.get_scan_reconstructions()[method]

        extents = recon.extent

        thb = data.get_scan_thb()

        if args.drawthb:
            recon = thb[(method, "0")]

        roi_drawerer = ROIDrawer(data, recon, extents, roi_name=args.name, roi_position=args.position,
                                 interpolation=args.interpolation)


if __name__ == "__main__":
    main()
