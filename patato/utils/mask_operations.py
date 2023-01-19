#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from __future__ import annotations

from typing import List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rois.roi_type import ROI

import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon


def to_binary_mask(vertices, min_x, max_x, nx, min_y, max_y, ny):
    from matplotlib import path
    if type(vertices) == np.ndarray or type(vertices) == list:
        vert_path = path.Path(vertices, closed=False)
    else:
        vert_path = vertices
    xs = np.linspace(min_x, max_x, nx)
    ys = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(xs, ys)
    points = np.array([X.flatten(), Y.flatten()]).T
    if type(vert_path) in [Polygon, MultiPolygon]:
        points = [Point(r) for r in points]
        mask = np.array([vert_path.contains(r) for r in points])#.reshape(X.shape)
    else:
        mask = vert_path.contains_points(points)
    return mask.reshape(X.shape)


def get_polygon_mask(p, fov_x, fov_y, nx, ny):
    if type(p) == Polygon:
        mask = to_binary_mask(np.array(p.exterior.coords.xy).T, -fov_x / 2, fov_x / 2, nx, -fov_y / 2, fov_y / 2,
                              ny)
        for interior in p.interiors:
            mask &= ~to_binary_mask(np.array(interior.coords.xy).T, -fov_x / 2, fov_x / 2, nx, -fov_y / 2, fov_y / 2,
                                    ny)
    elif type(p) == MultiPolygon:
        mask = get_polygon_mask(p.geoms[0], fov_x, fov_y, nx, ny)
        for g in p.geoms[1:]:
            mask |= get_polygon_mask(g, fov_x, fov_y, nx, ny)
    else:
        print("WARNING: something strange happening...")
        mask = np.zeros((nx, ny))
    return mask


def generate_mask(vertices, fov_x, nx, fov_y=None, ny=None):
    if fov_y is None:
        fov_y = fov_x
    if ny is None:
        ny = nx

    if type(vertices) in [Polygon, MultiPolygon]:
        mask = get_polygon_mask(vertices, fov_x, fov_y, nx, ny)
    else:
        mask = to_binary_mask(vertices, -fov_x / 2, fov_x / 2, nx, -fov_y / 2, fov_y / 2, ny)

    return mask


def interpolate_rois(rois: List["ROI"], z_positions):
    if len(rois) <= 1 or all([rois[0].z == r.z for r in rois]):
        return []
    from itk import MorphologicalContourInterpolator, image_view_from_array, array_view_from_image
    from .rois.roi_type import ROI
    buffer = 10

    indices = [r.ax0_index[0] for r in rois]

    min_index = min(indices)
    max_index = max(indices)

    zs = z_positions[min_index:max_index + 1, 0]
    points = [r.points for r in rois]
    x_0 = np.min([np.min(p, axis=0) for p in points], axis=0)
    x_1 = np.max([np.max(p, axis=0) for p in points], axis=0)
    dx = np.min((x_1 - x_0) / 200)

    nx = ((x_1 - x_0) / dx).astype(np.int32)
    minx = x_0 - buffer * dx
    maxx = x_1 + (buffer - 1) * dx
    nx = nx + 20

    mask = np.zeros((zs.shape[0], nx[1], nx[0]), np.uint16)

    for r, p in zip(rois, points):
        mask[r.ax0_index - min_index] = to_binary_mask(p, minx[0], maxx[0], nx[0],
                                                       minx[1], maxx[1], nx[1])
    mask = mask
    im = image_view_from_array(mask)
    interp = MorphologicalContourInterpolator(im)
    np_view = array_view_from_image(interp)
    import cv2 as cv
    shape = [np.squeeze(cv.findContours(i, 1, 2)[0][0].astype(float)) for i in np_view.astype(np.uint8)]

    paths = []
    for j, s in enumerate(shape):
        i = j + min_index
        if i not in indices:
            x, y = s.T

            x *= dx
            y *= -dx
            x += x_0[0] - 10.5 * dx
            y = -x_0[1] - y - nx[1] * dx + 10.5 * dx
            paths.append(ROI(np.array([x, -y]).T, zs[j], rois[0].run,
                             rois[0].repetition, rois[0].roi_class + "~interpolated",
                             rois[0].position, rois[0].generated, [i]))

    return paths
