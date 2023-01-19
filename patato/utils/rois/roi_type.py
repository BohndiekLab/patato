#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from typing import TYPE_CHECKING

import numpy as np
from shapely.validation import make_valid

from ...io.attribute_tags import ROITags
from ...utils.mask_operations import generate_mask

if TYPE_CHECKING:
    from typing import Dict
    from ...core.image_structures.image_sequence import ImageSequence
from shapely.geometry import Polygon, MultiPolygon


class ROI:
    """Class to store regions of interests.
    """
    def get_area(self):
        return self.get_polygon().area

    def get_polygon(self) -> Polygon:
        if type(self.points) in [Polygon, MultiPolygon]:
            if not self.points.is_valid:
                return make_valid(self.points)
            else:
                return self.points
        else:
            return make_valid(Polygon(self.points))

    @property
    def attributes(self) -> "Dict":
        output = {ROITags.Z_POSITION: self.z, ROITags.RUN: self.run, ROITags.REPETITION: self.repetition,
                  ROITags.ROI_NAME: self.roi_class, ROITags.ROI_POSITION: self.position,
                  ROITags.GENERATED_ROI: self.generated}
        return output

    def __init__(self, points, z_position, run, repetition, roi_class,
                 position, generated=False, ax0_index=None):
        self.points = points
        self.z = z_position
        self.run = run
        self.repetition = repetition
        self.roi_class = roi_class
        self.position = position
        self.generated = generated
        self.ax0_index = np.array(ax0_index)

    def to_mask_slice(self, image: "ImageSequence", return_selection=False):
        mask = generate_mask(self.points, image.fov[0], image.shape_2d[-1],
                             image.fov[1], image.shape_2d[-2])
        mask = mask.reshape(image.shape[-image.n_im_dim:])
        selection = slice(None, None)
        if image.ax_0_exists():
            selection = np.where(self.ax0_index[None, :] == np.atleast_1d(image.ax_0_labels)[:, None])[0]
            ret_image = image[selection]
        else:
            ret_image = image
        if not return_selection:
            return mask, ret_image
        else:
            return mask, ret_image, selection

    def plot(self, ax=None, **kwargs):
        from ..roi_operations import REGION_COLOUR_MAP, close_loop
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        if type(self.points) is np.ndarray:
            plot = ax.plot(close_loop(self.points)[:, 0], close_loop(self.points)[:, 1],
                           label=self.roi_class + "_" + self.position,
                           c=REGION_COLOUR_MAP[self.roi_class], **kwargs)
        else:
            x, y = self.points.exterior.coords.xy
            plot = ax.plot(x, y,
                           label=self.roi_class + "_" + self.position,
                           c=REGION_COLOUR_MAP[self.roi_class], **kwargs)
            for interior in self.points.interiors:
                x, y = interior.coords.xy
                plot.append(ax.plot(x, y,
                               label=self.roi_class + "_" + self.position,
                               c=REGION_COLOUR_MAP[self.roi_class], **kwargs))
        return plot
