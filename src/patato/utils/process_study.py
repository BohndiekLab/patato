#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import glob
from os.path import join

from patato.io.msot_data import PAData
from patato.utils import sort_key


def get_hdf5_files(folder, filter_name="", mode="r"):
    for file in sorted(glob.glob(join(folder, "**", "*.hdf5"), recursive=True), key=sort_key):
        pa_data = PAData.from_hdf5(file, mode)
        if filter_name.lower() not in pa_data.get_scan_name().lower() and filter_name.lower() not in file.lower():
            pa_data.scan_reader.close()
            continue
        else:
            yield file, pa_data
