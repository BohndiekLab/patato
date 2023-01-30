#  Copyright (c) Thomas Else 2023.
#  License: MIT

import os
import shutil
import zipfile
from tempfile import mkdtemp
from tqdm.auto import tqdm

import requests
from patato import iTheraMSOT

from .. import PAData

def get_patato_data_folder():
    """Get the folder where patato data is stored.

    Returns
    -------
    folder : str
        The folder where patato data is stored.
    """
    if os.environ.get("PAT_DATA_FOLDER", None) == "TEMP":
        os.environ["PAT_DATA_FOLDER"] = mkdtemp()
    return os.path.expanduser(os.environ.get("PAT_DATA_FOLDER", "~/patato_example_data"))


def download_file(file_from, file_to):
    print(f"Downloading {file_from} to {file_to}... Might take a while.")
    with requests.get(file_from, stream=True) as file_data:
        # Get the total file size
        total_length = int(file_data.headers.get("Content-Length"))
        # Download the file
        with tqdm.wrapattr(file_data.raw, "read", total=total_length, desc="") as raw:
            with open(file_to, "wb") as output:
                shutil.copyfileobj(raw, output)


def get_msot_time_series_example(image_type="so2"):
    """Get a time series of MSOT images.

    Returns
    -------
    dataset : PAData
        The MSOT dataset.
    """

    data_sources = {"so2": "https://www.repository.cam.ac.uk/bitstream/handle/1810/345836/invivo_oe.hdf5?sequence=6&isAllowed=y",
                    "icg": "https://www.repository.cam.ac.uk/bitstream/handle/1810/345836/invivo_dce.hdf5?sequence=7&isAllowed=y"}

    data_path = os.path.join(get_patato_data_folder(), f'{image_type}-timeseries-data.hdf5')
    folder = os.path.split(data_path)[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(data_path):
        # Download the data
        download_file(data_sources[image_type], data_path)
    return PAData.from_hdf5(data_path)


def get_ithera_msot_time_series_example(image_type="so2"):
    """Get a time series of MSOT images in the iThera format.

    Returns
    -------
    dataset : PAData
        The MSOT dataset.
    """
    data_sources = {"so2": "https://www.repository.cam.ac.uk/bitstream/handle/1810/345836/ithera_invivo_oe.zip?sequence=4&isAllowed=y",
                    "icg": "https://www.repository.cam.ac.uk/bitstream/handle/1810/345836/ithera_invivo_dce.zip?sequence=5&isAllowed=y"}

    data_path = os.path.join(get_patato_data_folder(), f'{image_type}-ithera_data')
    filenames = {"so2": "Scan_9", "icg": "Scan_10"}
    folder = os.path.split(data_path)[0]
    if folder and not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(data_path):
        # Download the data
        zip_file = os.path.join(mkdtemp(), "patato_temp.zip")
        download_file(data_sources[image_type], zip_file)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    return PAData(iTheraMSOT(os.path.join(data_path, filenames[image_type])))


def get_msot_phantom_example(image_type="clinical"):
    """Get a time series of MSOT images.

    Returns
    -------
    dataset : PAData
        The MSOT dataset.
    """
    data_sources = {"clinical": "https://www.repository.cam.ac.uk/bitstream/handle/1810/345836/clinical_phantom.hdf5?sequence=8&isAllowed=y",
                    "preclinical": "hhttps://www.repository.cam.ac.uk/bitstream/handle/1810/345836/preclinical_phantom.hdf5?sequence=3&isAllowed=y"}

    data_path = os.path.join(get_patato_data_folder(), f'{image_type}-msot-data.hdf5')
    folder = os.path.split(data_path)[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(data_path):
        # Download the data
        download_file(data_sources[image_type], data_path)
    return PAData.from_hdf5(data_path)
