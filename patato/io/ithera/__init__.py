#  Copyright (c) Thomas Else 2023.
#  License: MIT

import numpy as np


def load_ithera_irf(irf_file):
    irf_data = np.fromfile(irf_file, dtype=np.double)
    if irf_data[0] >= 1000:
        irf_data[0] -= 1000
    return irf_data
