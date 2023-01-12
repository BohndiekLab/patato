#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import os
import gdown
from .. import PAData


def get_patato_data_folder():
    """Get the folder where patato data is stored.

    Returns
    -------
    folder : str
        The folder where patato data is stored.
    """
    return os.path.expanduser(os.environ.get("PAT_DATA_FOLDER", "~/patato_example_data"))


def get_msot_time_series_example():
    """Get a time series of MSOT images.

    Returns
    -------
    dataset : PAData
        The MSOT dataset.
    """
    data_path = os.path.join(get_patato_data_folder(), 'timeseriesdata.hdf5')
    folder = os.path.split(data_path)[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(data_path):
        # Download the data
        gdown.download("https://drive.google.com/uc?id=1jeoEtd_8EpA3XTSF5J_eUCJQaXvvpxbR",
                       data_path, quiet=False)
    return PAData.from_hdf5(data_path)
