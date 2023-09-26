import glob
from pathlib import Path

import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, Listbox, messagebox
import customtkinter as ctk
import numpy as np
from matplotlib.widgets import PolygonSelector
from .. import PAData, ROI
from ..utils.roi_operations import ROI_NAMES, REGION_COLOURS, close_loop

import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from importlib.resources import files

POSITIONS = ["", "left", "right", "full", "radial", "ulnar", "superficial"]


def shorten(text, width, placeholder=""):
    if len(text) > width:
        return text[:width] + placeholder
    else:
        return text


class HDF5ViewerApp:
    def __init__(self, root, start_file=None):
        self.new_roi_vertices = None
        self.pa_data_selected = None
        self.polygon_selector = None
        self.recon_map = {}

        self.root = root
        self.root.title("PATATO: Draw Region of Interest")

        # Set grid weights for columns and rows to allow resizing
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)

        self.create_widgets()  # Create the initial widgets

        self.regions = {}
        self.region_names = []
        self.drawing = False

        if start_file is not None:
            self.load_hdf5_folder(start_file)

    def roi_listbox_focusout(self, x):
        self.roi_listbox.selection_clear(0, tk.END)
        self.button3.configure(state="disabled")
        self.button4.configure(state="disabled")
        for r in range(len(self.region_names)):
            line = self.regions[self.region_names[r]]
            if line is not None:
                line[0].set_linestyle("solid")
                line[0].set_zorder(1)
        self.canvas.draw()

    def roi_listbox_focusin(self, x):
        sel = self.roi_listbox.curselection()
        if sel:
            self.button3.configure(state="normal")
            self.button4.configure(state="normal")
            for r in range(len(self.region_names)):
                line = self.regions[self.region_names[r]]
                if r == sel[0]:
                    if line is not None:
                        line[0].set_linestyle("dashed")
                        line[0].set_zorder(100)
                else:
                    if line is not None:
                        line[0].set_linestyle("solid")
                        line[0].set_zorder(1)
            self.canvas.draw()

    def create_widgets(self):
        # Create a label to display the loaded HDF5 file path
        self.file_label = ctk.CTkLabel(self.root, text="No file loaded.")
        self.file_label.grid(row=0, column=0, columnspan=3, pady=5)

        # Create a button to load a folder containing HDF5 files
        self.load_button = ctk.CTkButton(self.root, text="Load HDF5 Folder", command=self.load_hdf5_folder)
        self.load_button.grid(row=1, column=0, columnspan=3, pady=5)

        # Create a frame for additional widgets on the left
        self.left_frame = ctk.CTkFrame(self.root, width=125)
        self.left_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        # Create a title label for the Listbox
        self.listbox_title_label = ctk.CTkLabel(self.left_frame, text="HDF5 Files")
        self.listbox_title_label.pack()

        # Create a listbox to display the list of HDF5 files
        self.file_listbox = Listbox(self.left_frame)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self.load_selected_hdf5_file)

        # Create a frame for Matplotlib and navigation toolbar in the center
        self.middle_frame = ctk.CTkFrame(self.root, width=200)
        self.middle_frame.grid(row=2, column=1, sticky="nsew")

        # Create a frame for additional widgets on the right
        self.right_frame = ctk.CTkFrame(self.root, width=125)
        self.right_frame.grid(row=2, column=2, padx=5, pady=5, sticky="nsew")

        # Create a Matplotlib figure and canvas for displaying the image dataset
        dpi = 75
        self.fig, self.ax = plt.subplots(figsize=(400 / dpi, 400 / dpi), dpi=dpi)

        # Hack to fix the icon.
        data = files('patato.convenience_scripts').joinpath('PATATOLogo.png').read_bytes()
        icon_image = tk.PhotoImage(data=data)
        self.root.iconphoto(True, icon_image)

        self.fig.set_facecolor((0.,) * 4)
        self.ax.set_facecolor((0.,) * 4)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.middle_frame)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        toolbar = NavigationToolbar2Tk(self.canvas, self.middle_frame)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create sliders
        self.slider1_label = ctk.CTkLabel(self.right_frame, text="Frame Number:")
        self.slider1 = ctk.CTkSlider(self.right_frame, from_=0, to=1, number_of_steps=1, command=self.update_slider)
        self.slider1.set(0)
        self.slider2_label = ctk.CTkLabel(self.right_frame, text="Wavelength Number:")
        self.slider2 = ctk.CTkSlider(self.right_frame, from_=0, to=1, number_of_steps=1, command=self.update_slider)
        self.slider2.set(0)

        # Create buttons
        self.button1 = ctk.CTkButton(self.right_frame, text="Draw ROI", command=self.callback_button1)
        self.button2 = ctk.CTkButton(self.right_frame, text="Save ROI", command=self.callback_button2, state="disabled")
        self.button3 = ctk.CTkButton(self.right_frame, text="Delete ROI",
                                     command=self.callback_button3, state="disabled")
        self.button4 = ctk.CTkButton(self.right_frame, text="Rename ROI",
                                     command=self.callback_button4, state="disabled")

        # Create a listbox for "roi" items
        self.roi_listbox = Listbox(self.right_frame, selectmode=tk.SINGLE)

        self.roi_listbox.bind('<FocusOut>', self.roi_listbox_focusout)
        self.roi_listbox.bind('<<ListboxSelect>>', self.roi_listbox_focusin)

        # Create a reconstruction methods dropdown
        self.recon_dropdown = ctk.CTkOptionMenu(self.right_frame, values=["Reconstruction Methods"],
                                                command=self.change_reconstruction)

        # Create a dropdown box with a button
        self.dropdown_label = ctk.CTkLabel(self.right_frame, text="Select ROI Name and Position:")
        self.roi_listbox_label = ctk.CTkLabel(self.right_frame, text="Existing ROIs:")
        self.roi_name_dropdown = ctk.CTkOptionMenu(self.right_frame, values=ROI_NAMES)
        self.roi_position_dropdown = ctk.CTkOptionMenu(self.right_frame, values=POSITIONS)

        # Place widgets using grid layout
        self.roi_listbox_label.grid(row=0, column=0, padx=5, pady=5)
        self.roi_listbox.grid(row=1, column=0, padx=5, pady=5)

        self.slider1_label.grid(row=2, column=0, padx=5, pady=5)
        self.slider1.grid(row=3, column=0, padx=5, pady=5)
        self.slider2_label.grid(row=4, column=0, padx=5, pady=5)
        self.slider2.grid(row=5, column=0, padx=5, pady=5)

        self.recon_dropdown.grid(row=6, column=0, padx=5, pady=5)

        self.button1.grid(row=7, column=0, padx=5, pady=5)
        self.button2.grid(row=8, column=0, padx=5, pady=5)
        self.button3.grid(row=9, column=0, padx=5, pady=5)
        self.button4.grid(row=10, column=0, padx=5, pady=5)

        self.dropdown_label.grid(row=11, column=0, padx=5, pady=5)
        self.roi_name_dropdown.grid(row=12, column=0, padx=5, pady=5)
        self.roi_position_dropdown.grid(row=13, column=0, padx=5, pady=5)

        self.hdf5_files = []  # List to store HDF5 file paths
        self.scan_names = []  # List to store scan names

    def update_slider(self, e):
        self.update_image()

    def load_hdf5_folder(self, folder_path=None):
        if folder_path is None:
            folder_path = filedialog.askdirectory(
                title="Please select a folder containing PATATO HDF5 files to analyse.")
        if folder_path:
            # Clear the listbox, the list of HDF5 files, and the list of scan names
            self.file_listbox.delete(0, tk.END)
            self.hdf5_files.clear()
            self.scan_names.clear()

            # Scan the folder for HDF5 files and read the 'scan_name' attribute
            i = 0

            files = glob.glob(os.path.join(folder_path, "**/*.hdf5"), recursive=True)

            for file_path in sorted(files, key=lambda x: PAData.from_hdf5(x).get_scan_datetime()):
                i += 1
                if i > 1000:
                    self.show_error_message("Too many files in the selected folder. Please choose a different "
                                            "folder.")
                    return

                self.hdf5_files.append(file_path)

                pa_data = PAData.from_hdf5(file_path, 'r')
                scan_name = shorten(Path(file_path).stem, width=8,
                                    placeholder="...") + ": " + pa_data.get_scan_name()

                self.scan_names.append(scan_name)
                self.file_listbox.insert(tk.END, scan_name)
                pa_data.scan_reader.file.close()

            if self.hdf5_files:
                self.sort_file_list()  # Sort the list based on the custom sort key
                self.show_selected_hdf5_file(0)
                self.file_label.configure(text=f"Loaded Folder: {folder_path}")
            else:
                self.show_error_message("No HDF5 files found in the selected folder.")
        else:
            self.show_error_message("Please choose a folder to load data from.")

    def load_selected_hdf5_file(self, event):
        selected_index = self.file_listbox.curselection()
        if selected_index:
            index = int(selected_index[0])
            self.show_selected_hdf5_file(index)

    def update_image(self):
        i = int(self.slider1.get())
        j = int(self.slider2.get())
        # Display the image dataset using Matplotlib
        self.ax.clear()

        recon_method = self.recon_dropdown.get()
        if recon_method == "Ultrasound":
            recon = self.pa_data_selected.get_ultrasound()
        else:
            recon = self.pa_data_selected.get_scan_reconstructions()[self.recon_map[recon_method]]
        recon = recon[i, j]
        recon.imshow(ax=self.ax)

        self.regions = {}
        rois = list(self.pa_data_selected.get_rois().keys())
        self.region_names = rois

        rois = self.pa_data_selected.get_rois()
        # Plot the ROIS
        for (r, n) in rois:
            draw_all_rois = False  # Update this to a check-box
            clinical = self.pa_data_selected.is_clinical()
            frame_type = "z" if not clinical else "repetition"
            match_frames = self.pa_data_selected.get_z_positions() if not clinical else self.pa_data_selected.get_repetition_numbers()
            if np.isclose(match_frames[i, j],
                          rois[r, n].attributes.get(frame_type, 1.0)) or draw_all_rois:
                colour = "C0"
                for k, c in zip(ROI_NAMES, REGION_COLOURS):
                    if k in r:
                        colour = c

                roi_points = rois[r, n].points
                roi_close = close_loop(roi_points)
                # TODO: Control the slice through the image to draw.
                roi_plot = self.ax.plot(roi_close[:, 0], roi_close[:, 1], picker=True,
                                        label=r + "/" + n, c=colour, scalex=False,
                                        scaley=False)
            else:
                roi_plot = None
            self.regions[r, n] = roi_plot
        self.populate_roi_listbox(self.region_names)
        self.canvas.draw()

    def change_reconstruction(self, x):
        self.update_image()

    def show_selected_hdf5_file(self, index):
        if 0 <= index < len(self.hdf5_files):
            self.button3.configure(state="disabled")
            self.button4.configure(state="disabled")
            file_path = self.hdf5_files[index]
            try:
                self.file_label.configure(text=f"Loaded File: {file_path}")

                # Open the selected HDF5 file
                pa_data = PAData.from_hdf5(file_path, "r+")
                self.pa_data_selected = pa_data

                self.recon_map = {r_name[0] + f"({r_name[1]})": r_name for r_name in pa_data.get_scan_reconstructions()}
                recon_methods = list(self.recon_map.keys())
                if pa_data.get_ultrasound():
                    recon_methods.append("Ultrasound")
                self.recon_dropdown.configure(values=recon_methods)
                if self.recon_dropdown.get() not in recon_methods:
                    self.recon_dropdown.set(recon_methods[0])

                if pa_data.get_scan_reconstructions():
                    recon = pa_data.get_scan_reconstructions()

                    # Change this so that you can view different reconstructions
                    recon = recon[list(recon.keys())[0]]

                    # Update the frame and wavelength sliders.
                    self.slider1.configure(from_=0, to=recon.shape[0] - 1, number_of_steps=recon.shape[0] - 1)
                    if recon.shape[0] == 1:
                        self.slider1.configure(state="disabled")
                    else:
                        self.slider1.set(0)
                        self.slider1.configure(state="normal")
                    self.slider2.configure(from_=0, to=recon.shape[1] - 1, number_of_steps=recon.shape[1] - 1)
                    if recon.shape[1] == 1:
                        self.slider2.configure(state="disabled")
                    else:
                        self.slider2.set(0)
                        self.slider2.configure(state="normal")
                    self.slider2.set(0)

                    # Populate the ROI listbox with items from the 'roi' group
                    self.update_image()
                else:
                    self.show_error_message(f"No reconstructions available.")
            except Exception as e:
                import traceback
                self.show_error_message(f"Error loading HDF5 file: {str(e)}")
                traceback.print_exception(type(e), e, e.__traceback__)

    def show_error_message(self, message):
        error_window = ctk.CTkToplevel(self.root)
        error_window.title("Error")

        error_label = ctk.CTkLabel(error_window, text=message)
        error_label.pack(padx=20, pady=20)

    def sort_file_list(self):
        # Clear the listbox and insert the sorted scan names
        self.file_listbox.delete(0, tk.END)
        for scan_name in self.scan_names:
            self.file_listbox.insert(tk.END, scan_name)

    def populate_roi_listbox(self, labels):
        # Clear the ROI listbox
        self.roi_listbox.configure(state="normal")
        self.roi_listbox.delete(0, tk.END)
        for roi_name in labels:
            self.roi_listbox.insert(tk.END, roi_name)
        self.roi_listbox_focusout(None)

    def toggle_drawing(self):
        widgets = [self.load_button,
                   self.roi_listbox, self.file_listbox
                   ]
        widgets_opposite = [self.button2]
        if self.drawing:
            for w in widgets:
                w.configure(state="normal")
            for w in widgets_opposite:
                w.configure(state="disabled")
            self.button1.configure(text="Draw ROI")
            # Stop drawing, enable tools
        else:
            for w in widgets:
                w.configure(state="disabled")
            for w in widgets_opposite:
                w.configure(state="normal")
            self.button1.configure(text="Cancel ROI")
            # Start drawing, disable tools
        self.drawing = not self.drawing

    def poly_onselect(self, v):
        self.new_roi_vertices = v

    # Callback functions for buttons and dropdown
    def callback_button1(self):
        # Start drawing regions of interest
        self.roi_listbox_focusout(None)
        self.toggle_drawing()
        if self.drawing:
            self.polygon_selector = PolygonSelector(self.ax,
                                                    self.poly_onselect,
                                                    props=dict(color="r")
                                                    )
        else:
            self.polygon_selector.set_visible(False)
            self.polygon_selector = None
            self.canvas.draw()

    def callback_button2(self):
        i = int(self.slider1.get())
        j = int(self.slider2.get())
        z = self.pa_data_selected.get_z_positions()[i, j]
        run = self.pa_data_selected.get_run_number()[i, j]
        repetition = self.pa_data_selected.get_repetition_numbers()[i, j]
        roi_name = self.roi_name_dropdown.get()
        roi_position = self.roi_position_dropdown.get()
        self.pa_data_selected.add_roi(ROI(self.new_roi_vertices, z, run, repetition, roi_name, roi_position))
        self.update_image()
        # Disable drawing at the end
        self.callback_button1()

    def callback_button3(self):
        sel = self.roi_listbox.curselection()
        answer = False
        if sel:
            answer = messagebox.askyesno("Question", "Are you sure you want to delete this ROI?")
        if answer and sel:
            self.pa_data_selected.delete_rois(*self.region_names[sel[0]])
        self.button3.configure(state="disabled")
        self.button4.configure(state="disabled")
        self.update_image()
        self.roi_listbox_focusout(None)

    def callback_button4(self):
        sel = self.roi_listbox.curselection()
        answer = False
        if sel:
            answer = messagebox.askyesno("Question", "Are you sure you want to rename this ROI?")
        if answer and sel:
            roi_name = self.roi_name_dropdown.get()
            roi_position = self.roi_position_dropdown.get()
            self.pa_data_selected.rename_roi(self.region_names[sel[0]], roi_name, roi_position)
        self.button3.configure(state="disabled")
        self.button4.configure(state="disabled")
        self.update_image()
        self.roi_listbox_focusout(None)


def main():
    plt.ioff()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("dark-blue")
    ctk.DrawEngine.preferred_drawing_method = "circle_shapes"
    root = ctk.CTk(className='PATATO', baseName="PATATO")
    data = files('patato.convenience_scripts').joinpath('PATATOLogo.png').read_bytes()
    icon_image = tk.PhotoImage(data=data)
    # Set the icon for the main window
    root.iconphoto(True, icon_image)

    root.title("PATATO: Draw Region of Interest. Please select a folder.")
    app = HDF5ViewerApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
    root.update()

    answer = messagebox.askokcancel("Info", "Please choose a folder containing PATATO HDF5 files to analyse.")
    if not answer:
        root.quit()
        root.destroy()
        return
    app.load_hdf5_folder()

    root.mainloop()


if __name__ == "__main__":
    print("Running main")
    main()
