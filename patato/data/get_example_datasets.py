#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import os
import zipfile
from tempfile import mkdtemp

import gdown
from patato import iTheraMSOT

from .. import PAData


def get_patato_data_folder():
    """Get the folder where patato data is stored.

    Returns
    -------
    folder : str
        The folder where patato data is stored.
    """
    return os.path.expanduser(os.environ.get("PAT_DATA_FOLDER", "~/patato_example_data"))


def get_msot_time_series_example(image_type="so2"):
    """Get a time series of MSOT images.

    Returns
    -------
    dataset : PAData
        The MSOT dataset.
    """
    data_sources = {"so2": "https://drive.google.com/uc?id=1la0i2qEpg_80Q92ScWVF-H_DNfV87Vzh",
                    "icg": "https://drive.google.com/uc?id=1lZF4VMlnbreDIQSixj0LBfRgd9JlqhUR"}

    data_path = os.path.join(get_patato_data_folder(), f'{image_type}-timeseries-data.hdf5')
    folder = os.path.split(data_path)[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(data_path):
        # Download the data
        gdown.download(data_sources[image_type],
                       data_path, quiet=False)
    return PAData.from_hdf5(data_path)


def get_ithera_msot_time_series_example(image_type="so2"):
    """Get a time series of MSOT images in the iThera format.

    Returns
    -------
    dataset : PAData
        The MSOT dataset.
    """
    data_sources = {"so2": "https://drive.google.com/uc?id=1lhO81fxcq1VUQ3H2FeHtcTujADuZwDyQ",
                    "icg": "https://drive.google.com/uc?id=1lctq09X6xrCC33usr5SoIkhJJByYZuRt"}

    data_path = os.path.join(get_patato_data_folder(), f'{image_type}-ithera_data')
    filenames = {"so2": "Scan_9", "icg": "Scan_10"}
    folder = os.path.split(data_path)[0]
    if folder and not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(data_path):
        # Download the data
        f = gdown.download(data_sources[image_type], os.path.join(mkdtemp(), "patato_temp.zip"), quiet=False)
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    return PAData(iTheraMSOT(os.path.join(data_path, filenames[image_type])))


def get_msot_phantom_example(image_type="clinical"):
    """Get a time series of MSOT images.

    Returns
    -------
    dataset : PAData
        The MSOT dataset.
    """
    data_sources = {"clinical": "",
                    "preclinical": ""}

    data_path = os.path.join(get_patato_data_folder(), f'{image_type}-msot-data.hdf5')
    folder = os.path.split(data_path)[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(data_path):
        # Download the data
        gdown.download(data_sources[image_type],
                       data_path, quiet=False)
    return PAData.from_hdf5(data_path)

