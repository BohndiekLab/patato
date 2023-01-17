#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from typing import Optional

import numpy as np
import pandas as pd

from ..utils.process_study import get_hdf5_files


def process_scan_name(template: str, scan_name: str) -> dict:
    """
    Process the scan name using the simple template specified.

    The template should contain something like the following:

    >>> "<date>_<initials><earmark><mouseid>_Day<timepoint>_<scantype>"

    For advanced use, note that this is converted into a regular expression,
    so certain elements of that syntax can be used in the template.

    Parameters
    ----------
    template
    scan_name

    Returns
    -------

    """
    import re
    regex_codes = {"Date": r"([0-9]*)",
                   "Initials": r"([A-z]{2,3})",
                   "EarMark": r"(NM|1L|1R|2L|2R|1L1R|1R1L|1RL|1B|IB|IL|IR)",
                   "MouseID": r"([0-9]{1,6})",
                   "ScanType": r"([A-z|0-9|_|\+]+)?",
                   "Timepoint": r"([0-9]+)"}

    # template = template.replace("(", "(?:")
    for k, code in regex_codes.items():
        code = f"(?P<{k}>" + code[1:]
        template = template.replace(f"<{k}>", code)

    p = re.compile(template)
    try:
        d = p.match(scan_name).groupdict()
        for k, v in d.items():
            if type(v) == str:
                d[k] = v.upper()
        return d
    except AttributeError:
        print(f"Unable to match template to scan name: {scan_name}.")
        return {}


def invert_dictionary_tolist(mapping):
    return {mouse: date for date, mice in mapping.items() for mouse in mice}


def extract_data_tables(datafolder: str, name_template: str,
                        analyse_rois: list, metrics=None,
                        start_days=None,
                        group_info: Optional[dict] = None,
                        reconstruction_name=None,
                        analyse_scan_types=None,
                        just_summary=True,
                        roi_kwargs=None,
                        apply_function=None,
                        filter_name="",
                        more_details=None,
                        roi_source_type=None,
                        return_masks=False
                        ):
    if start_days is None:
        start_days = {}
    if group_info is None:
        group_info = {}
    if metrics is None:
        metrics = ["thb", "so2"]

    start_date_map = {timepoint: invert_dictionary_tolist(mapping) for timepoint, mapping in start_days.items()}
    group_info = {timepoint: invert_dictionary_tolist(mapping) for timepoint, mapping in group_info.items()}

    images = []
    tables = []

    # Share regions of interest between adjacent scans.
    datasets = list(get_hdf5_files(datafolder, filter_name=filter_name))
    dataset_details = [(process_scan_name(name_template, data.get_scan_name()), data) for _, data in datasets]

    # Generate a dictionary to lookup all the scans for each scan session. Using the MouseID and Timepoint as an ID.
    scan_map = {}
    for details, data in dataset_details:
        scan_id = (details.get("MouseID", None), details.get("Timepoint", None))
        if scan_id not in scan_map:
            scan_map[scan_id] = {}
        scan_map[scan_id][details.get("ScanType")] = data

    if roi_source_type is not None:
        for scan_id in scan_map:
            if roi_source_type not in scan_map[scan_id]:
                continue
            for scan_type in scan_map[scan_id]:
                scan_map[scan_id][scan_type].external_roi_interface = scan_map[scan_id][roi_source_type]

    # Loop through all datasets and extract data.
    for f, data in datasets:
        print(f, data.get_scan_name())
        # Set the default reconstruction method.
        data.set_default_recon(reconstruction_name)

        # Extract useful information from the scan name
        scan_name = str.strip(data.get_scan_name())
        details = process_scan_name(name_template, scan_name)

        if analyse_scan_types is not None:
            if details.get("ScanType", None) not in analyse_scan_types:
                continue

        if not details:
            continue
        details["File"] = f
        # Mouse id - must be set for this analysis code.
        mouse_id = int(details["MouseID"])

        # Get the scan date.
        date = data.get_scan_datetime()

        # Extract details (e.g treatment, cell line etc)
        for detail, mouse_mapping in group_info.items():
            details[detail] = mouse_mapping[mouse_id]

        # Extract the time data (e.g. time since dosing started, time since implantation)
        for time_detail, mouse_mapping in start_date_map.items():
            details[time_detail] = (date - mouse_mapping[mouse_id]).days + 1

        if more_details is not None:
            for fn in more_details:
                for k, v in fn(data).items():
                    details[k] = v

        if not data.get_rois():
            continue
        else:
            measurements = data.summary_measurements(metrics=metrics,
                                                     include_rois=analyse_rois,
                                                     roi_kwargs=roi_kwargs,
                                                     just_summary=just_summary,
                                                     return_masks=return_masks)
            for d in details:
                measurements[d] = details[d]
            measurements["Date"] = data.get_scan_datetime()
            if apply_function is not None:
                measurements = apply_function(measurements)
            tables.append(measurements)

    df = tables[0].append(tables[1:]).reset_index()

    df["Radius"] = np.sqrt(df["Area"] / np.pi) * 75 * 3e-3

    df["Volume"] = df["Radius"] ** 3 * 4 * np.pi / 3

    df["Date"] = pd.to_datetime(df["Date"], utc=True, dayfirst=True).dt.tz_localize(None)

    return df, images


def set_matplotlib_defaults(fig_width=91.5, fig_height=89):
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.sans-serif"] = "Arial"
    matplotlib.rcParams["figure.dpi"] = 227
    matplotlib.rcParams["figure.figsize"] = (
        fig_width / 25.4, fig_height / 25.4)  # OR 183 mm for double width
    matplotlib.rcParams["font.size"] = 7
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = False
    matplotlib.rcParams["savefig.pad_inches"] = 0
    matplotlib.rcParams['figure.subplot.bottom'] = 0.075
    matplotlib.rcParams['figure.subplot.hspace'] = 0.4
    matplotlib.rcParams['figure.subplot.left'] = 0.075
    matplotlib.rcParams['figure.subplot.right'] = 0.97
    matplotlib.rcParams['figure.subplot.top'] = 0.925
    matplotlib.rcParams['figure.subplot.wspace'] = 0.5
    matplotlib.rcParams["figure.titlesize"] = "medium"
    matplotlib.rcParams['axes.titlesize'] = "small"
    matplotlib.rcParams["lines.markersize"] = 3
    matplotlib.rcParams["legend.frameon"] = False
