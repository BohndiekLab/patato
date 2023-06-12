#  Copyright (c) Thomas Else 2023.
#  License: MIT

import h5py
import numpy as np
from patato.data.get_example_datasets import get_patato_data_folder

import os


def add_example_reconstruction(rs, shape):
    """
    Generate a dummy dataset.

    Parameters
    ----------
    rs
    shape
    """
    dr = rs.create_group("Dummy Recon")
    recon = dr.create_dataset("0", data=np.random.random(shape))
    recon.attrs["RECONSTRUCTION_NX"] = 333
    recon.attrs["RECONSTRUCTION_NY"] = 1
    recon.attrs["RECONSTRUCTION_NZ"] = 333
    recon.attrs["RECONSTRUCTION_FIELD_OF_VIEW_X"] = 0.025
    recon.attrs["RECONSTRUCTION_FIELD_OF_VIEW_Y"] = 0.025
    recon.attrs["RECONSTRUCTION_FIELD_OF_VIEW_Z"] = 0.025
    return recon


def make_dummy_dataset():
    data_folder = os.path.join(get_patato_data_folder(), "test")
    os.makedirs(data_folder, exist_ok=True)
    file = os.path.join(data_folder, "Scan_1.hdf5")
    if os.path.exists(file):
        return
    f = h5py.File(file, "a")
    thetas = np.linspace(np.pi / 4, 7 * np.pi / 4, 256)
    geometry = np.array([np.cos(thetas), np.sin(thetas)]) * 0.04
    irf = np.zeros((2030,))
    irf[1015] = 1
    wavelengths = np.array([650, 800])
    n_wavelengths = wavelengths.size
    n_runs = 20
    f.attrs["date"] = "2022-11-23T15:50:22+0000"
    ts = f.create_dataset("raw_data", data=np.random.random((n_runs, n_wavelengths, 256, 2030)) + 10, dtype=np.uint16)
    ts.attrs["fs"] = 4e7
    ts.attrs["name"] = "Demo Data"
    ts.attrs["speedofsound"] = 1500
    f.create_dataset("GEOMETRY", data=geometry.T)
    f.create_dataset("irf", data=irf)
    f.create_dataset("wavelengths", data=wavelengths)
    f.create_dataset("RUN", data=np.zeros((n_runs, n_wavelengths), dtype=np.int32))
    f.create_dataset("REPETITION", data=np.arange(n_runs)[:, None].repeat(n_wavelengths, axis=1))
    f.create_dataset("TEMPERATURE", data=30 * np.ones((n_runs, n_wavelengths)))
    f.create_dataset("OverallCorrectionFactor", data=np.ones((n_runs, n_wavelengths)))
    f.create_dataset("Z-POS", data=np.ones((n_runs, n_wavelengths)))
    f.create_dataset("timestamp",
                     data=np.arange(n_runs * n_wavelengths).reshape((n_runs, n_wavelengths)).astype(np.uint64))

    rs = f.create_group("recons")
    add_example_reconstruction(rs, (n_runs, n_wavelengths, 333, 1, 333))

    rs = f.create_group("unmixed")
    um = add_example_reconstruction(rs, (n_runs, 2, 333, 1, 333))
    um.attrs["SPECTRA"] = ["Hb", "HbO2"]
    um.attrs["WAVELENGTHS"] = wavelengths

    rs = f.create_group("so2")
    s = add_example_reconstruction(rs, (n_runs, 1, 333, 1, 333))
    s.attrs["SPECTRA"] = ["Hb", "HbO2"]
    s.attrs["WAVELENGTHS"] = wavelengths

    rs = f.create_group("thb")
    s = add_example_reconstruction(rs, (n_runs, 1, 333, 1, 333))
    s.attrs["SPECTRA"] = ["Hb", "HbO2"]
    s.attrs["WAVELENGTHS"] = wavelengths

    rs = f.create_group("dso2")
    dso2 = add_example_reconstruction(rs, (333, 1, 333))
    dso2.attrs.update(**s.attrs)

    rs = f.create_group("baseline_so2")
    dso2 = add_example_reconstruction(rs, (333, 1, 333))
    dso2.attrs.update(**s.attrs)

    rs = f.create_group("baseline_so2_sigma")
    dso2 = add_example_reconstruction(rs, (333, 1, 333))
    dso2.attrs.update(**s.attrs)

    rs = f.create_group("dicg")
    dso2 = add_example_reconstruction(rs, (333, 1, 333))
    dso2.attrs.update(**s.attrs)

    rs = f.create_group("baseline_icg")
    dso2 = add_example_reconstruction(rs, (333, 1, 333))
    dso2.attrs.update(**s.attrs)

    rs = f.create_group("baseline_icg_sigma")
    dso2 = add_example_reconstruction(rs, (333, 1, 333))
    dso2.attrs.update(**s.attrs)

    rois = f.create_group("rois")
    r = rois.create_group("tumour_left")
    roi = np.array([[0.01, 0.01], [0.01, -0.01], [-0.01, -0.01], [-0.01, 0.01]])
    rn = r.create_dataset("0", data=roi)
    rn.attrs["position"] = "left"
    rn.attrs["class"] = "tumour"
    rn.attrs["z"] = 1.0
    rn.attrs["run"] = 0
