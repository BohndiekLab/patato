#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from os.path import isfile

import unittest
import os
import sys

# make sure that we're using the downloaded version of patato.
sys.path.insert(0, os.path.abspath('../'))

import h5py
import numpy as np
from coverage import Coverage

cov = Coverage(source=['../patato'], omit=["*test*", "*convenience_scripts*", "*useful_utilities*"])
cov.start()

# noinspection PyPep8
from patato.processing.tests.preprocessing_algorithm import TestPreprocessing
from patato.recon.tests.backprojection_test import BackprojectionTest
from patato.unmixing.tests.unmixing_test import TestUnmixing
from patato.utils.tests.pipline_tests import TestPipelines
from test_image_sequence import TestHDF5Load
from patato.io.ithera.tests.ithera_tests import TestITheraImport
from patato.io.json.tests.test_reconstruction_reading import TestJSONLoading
from patato.io.tests.test_msot_data import TestMSOTData


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


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


if not isfile("../data/Scan_1.hdf5"):
    f = h5py.File("../data/Scan_1.hdf5", "a")
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

alltests = unittest.TestSuite()
alltests.addTest(unittest.makeSuite(TestMSOTData))
alltests.addTest(unittest.makeSuite(TestPreprocessing))
alltests.addTest(unittest.makeSuite(TestITheraImport))
alltests.addTest(unittest.makeSuite(BackprojectionTest))
alltests.addTest(unittest.makeSuite(TestUnmixing))
alltests.addTest(unittest.makeSuite(TestPipelines))
alltests.addTest(unittest.makeSuite(TestHDF5Load))
alltests.addTest(unittest.makeSuite(TestJSONLoading))

result = unittest.TextTestRunner(verbosity=2).run(alltests)

cov.stop()
cov.save()

cov.report(skip_empty=True, skip_covered=False)
cov.html_report(directory="../docs/test_coverage")
cov.xml_report(outfile="../docs/test_coverage/coverage.xml")
