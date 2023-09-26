import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, Listbox, messagebox
import customtkinter as ctk

from .. import PAData
from ..recon import get_default_recon_preset
from ..io.json.json_reading import read_reconstruction_preset

import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from importlib.resources import files


def shorten(text, width, placeholder=""):
    if len(text) > width:
        return text[:width] + placeholder
    else:
        return text


class HDF5ViewerApp:
    def update_speed_of_sound(self, x):
        c = self.slider3.get()
        self.speed_label.configure(text=f"{c} m/s")
        self.update_image()

    def save_speed_of_sound(self):
        self.pa_data_selected.set_speed_of_sound(self.slider3.get())

    def save_all_speed_of_sound(self):
        for f in self.hdf5_files:
            p = PAData.from_hdf5(f, "r+")
            p.set_speed_of_sound(self.slider3.get())

    def __init__(self, root, start_file=None):
        self.new_roi_vertices = None
        self.pa_data_selected = None
        self.polygon_selector = None
        self.recon_map = {}

        self.root = root
        self.root.title("PATATO: Set Speed of Sound")

        # Set grid weights for columns and rows to allow resizing
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=1)

        self.create_widgets()  # Create the initial widgets

        if start_file is not None:
            self.load_hdf5_folder(start_file)

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
        self.slider1 = ctk.CTkSlider(self.right_frame, from_=0, to=1, number_of_steps=1,
                                     command=self.update_slider)
        self.slider1.set(0)
        self.slider2_label = ctk.CTkLabel(self.right_frame, text="Wavelength Number:")
        self.slider2 = ctk.CTkSlider(self.right_frame, from_=0, to=1, number_of_steps=1,
                                     command=self.update_slider)
        self.slider2.set(0)

        speed = tk.IntVar(value=1500)
        self.slider3_label = ctk.CTkLabel(self.right_frame, text="Speed of sound (m/s)")
        self.slider3 = ctk.CTkSlider(self.right_frame, from_=1400, to=1600, number_of_steps=200,
                                     command=self.update_speed_of_sound, variable=speed)
        self.slider3.set(1500)

        self.speed_label = ctk.CTkLabel(self.right_frame, text="1500 m/s")

        self.cbutton = ctk.CTkButton(self.right_frame, text="Save Speed of Sound", command=self.save_speed_of_sound)
        self.cbutton_all = ctk.CTkButton(self.right_frame, text="Save Speed of Sound to All", command=self.save_all_speed_of_sound)

        self.slider1_label.grid(row=1, column=0, padx=5, pady=5)
        self.slider1.grid(row=2, column=0, padx=5, pady=5)
        self.slider2_label.grid(row=3, column=0, padx=5, pady=5)
        self.slider2.grid(row=4, column=0, padx=5, pady=5)

        self.slider3_label.grid(row=5, column=0, padx=5, pady=5)
        self.slider3.grid(row=6, column=0, padx=5, pady=5)
        self.cbutton.grid(row=7, column=0, padx=5, pady=5)
        self.speed_label.grid(row=8, column=0, padx=5, pady=5)
        self.cbutton_all.grid(row=9, column=0, padx=5, pady=5)

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

                preset = get_default_recon_preset(pa_data)

                with open(preset) as json_file:
                    settings = json.load(json_file)

                self.pipeline = read_reconstruction_preset(settings)
                self.pipeline.time_factor = 1
                self.pipeline.detector_factor = 1

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

        preprocessor = self.pipeline
        reconstructor = self.pipeline.children[0]

        pre_processed, recon_args, _ = preprocessor.run(self.pa_data_selected[i:i+1, j:j+1].get_time_series(),
                                                        self.pa_data_selected[i:i+1, j:j+1])

        recon_info, _, _ = reconstructor.run(pre_processed,
                                             self.pa_data_selected[i:i+1, j:j+1],
                                             speed_of_sound=self.slider3.get(), **recon_args)
        try:
            recon_info.imshow(ax=self.ax)
        except Exception as e:
            print(recon_info.shape, self.pa_data_selected.shape)
            print(e)
        self.canvas.draw()

    def change_reconstruction(self, x):
        self.update_image()

    def show_selected_hdf5_file(self, index):
        if 0 <= index < len(self.hdf5_files):
            file_path = self.hdf5_files[index]
            try:
                self.file_label.configure(text=f"Loaded File: {file_path}")

                # Open the selected HDF5 file
                pa_data = PAData.from_hdf5(file_path, "r+")
                self.pa_data_selected = pa_data

                if pa_data.get_speed_of_sound():
                    self.slider3.set(pa_data.get_speed_of_sound())
                    self.speed_label.configure(text=f"{pa_data.get_speed_of_sound()} m/s")

                # Update the frame and wavelength sliders.
                s = pa_data.shape
                self.slider1.configure(from_=0, to=s[0] - 1, number_of_steps=s[0] - 1)
                if s[0] == 1:
                    self.slider1.configure(state="disabled")
                else:
                    self.slider1.set(0)
                    self.slider1.configure(state="normal")
                self.slider2.configure(from_=0, to=s[1] - 1, number_of_steps=s[1] - 1)
                if s[1] == 1:
                    self.slider2.configure(state="disabled")
                else:
                    self.slider2.set(0)
                    self.slider2.configure(state="normal")
                self.slider2.set(0)

                self.update_image()
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

    root.title("PATATO: Set Speed of Sound. Please select a folder.")
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
