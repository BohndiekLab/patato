#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..io.msot_data import PAData

from seaborn import color_palette


def split_roi_left_right(data: "PAData", base_roi="", split_template="unnamed"):
    from ..io.msot_data import ROI
    # Initial implementation - only supports one "dividing" roi.
    data_rois = data.get_rois()

    template_data = None
    names = None
    if (split_template + "_left", "0") in data_rois:
        template_data = data_rois[(split_template + "_left", "0")]
        names = ["left", "right"]
    elif (split_template + "_right", "0") in data_rois:
        template_data = data_rois[(split_template + "_right", "0")]
        names = ["right", "left"]
    else:
        return []

    template_polygon = template_data.get_polygon()
    output = []
    for (name, number), roi in data_rois.items():
        if roi.position != "" or roi.generated:
            continue
        if base_roi in name:
            # Process
            base_polygon = roi.get_polygon()
            base_region_a = base_polygon & template_polygon
            base_region_b = base_polygon - base_region_a
            # Convert to numpy array
            try:
                x, y = base_region_a.exterior.coords.xy
                region_a = np.array([x[:-1], y[:-1]]).T
                x, y = base_region_b.exterior.coords.xy
                region_b = np.array([x[:-1], y[:-1]]).T
            except AttributeError:
                continue

            # Get the metadata.
            z_position = roi.z
            run = roi.run
            repetition = roi.repetition
            roi_class = roi.roi_class

            roi_a = ROI(region_a, z_position, run, repetition, roi_class,
                        names[0])
            roi_b = ROI(region_b, z_position, run, repetition, roi_class,
                        names[1])
            output.append(roi_a)
            output.append(roi_b)
    return output


ROI_NAMES = ["brain", "body", "reference", "aorta", "tumour", "background", "artery",
             "vein", "muscle", "phantom", "unnamed", "kidney", "spleen", "spine"]

REGION_COLOURS = color_palette("husl", len(ROI_NAMES))

REGION_COLOUR_MAP = defaultdict(lambda: REGION_COLOURS[-1])
for x, y in zip(ROI_NAMES, REGION_COLOURS):
    REGION_COLOUR_MAP[x] = y


def close_loop(x):
    return np.concatenate([x, x[0:1, :]])


def get_rim_core_rois(roi_data: "ROI", distance: float, radius=False):
    from ..io.msot_data import ROI
    roi = roi_data.get_polygon()
    if radius:
        effective_radius = np.sqrt(roi.area / np.pi)
        # distance is the desired radius
        distance = -distance + effective_radius

    roi_buffer = roi.buffer(-distance)

    core_roi = ROI(roi_buffer, roi_data.z, roi_data.run, roi_data.repetition, roi_data.roi_class + ".core",
                   roi_data.position, generated=True, ax0_index=roi_data.ax0_index)
    rim_roi = ROI(roi - roi_buffer, roi_data.z, roi_data.run, roi_data.repetition, roi_data.roi_class + ".rim",
                  roi_data.position, generated=True, ax0_index=roi_data.ax0_index)
    return core_roi, rim_roi


def add_rim_core_data(data: "PAData", base_roi: Tuple[str, str], distance: float, radius=False):
    roi_data: "ROI" = data.get_rois()[base_roi]
    return get_rim_core_rois(roi_data, distance, radius)
