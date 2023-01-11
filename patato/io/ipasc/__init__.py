#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import h5py
import pacfish as pf

from .ipasc_export import TomHDF5AdapterToIPASCFormat


def export_to_ipasc(path_to_hdf5, out_path=None):
    if out_path is None:
        out_path = path_to_hdf5.replace(".hdf5", "_ipasc.hdf5")

    hdf5_file = h5py.File(path_to_hdf5, "r")
    converter = TomHDF5AdapterToIPASCFormat(hdf5_file=hdf5_file)
    print(hdf5_file.keys())
    pa_data = converter.generate_pa_data()
    pf.write_data(out_path, pa_data)
