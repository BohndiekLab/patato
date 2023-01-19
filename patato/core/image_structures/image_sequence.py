"""
Image sequence - abstract classes for processing datasets from PA data.
"""

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Iterable

import numpy as np
import xarray

try:
    from pyopencl.array import Array
except ImportError:
    Array = None
try:
    import cupy as cp
except ImportError:
    cp = None

from xarray import DataArray
import jax.numpy as jnp
if jnp.DeviceArray not in xarray.core.variable.NON_NUMPY_SUPPORTED_ARRAY_TYPES:
    xarray.core.variable.NON_NUMPY_SUPPORTED_ARRAY_TYPES += (jnp.DeviceArray, )
from dask.array.core import Array as DaskArray

from ...io.attribute_tags import ReconAttributeTags
from ...processing.processing_algorithm import ProcessingResult
from ...utils.plotting import type_cmaps
from ...utils.rois.roi_type import ROI


def _get_matplotlib_scalebar_size(scalebar):
    # Mode 1: Auto
    ax = scalebar.axes
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    from matplotlib import rcParams

    def _get_value(attr, default):
        value = getattr(scalebar, attr)
        if value is None:
            value = rcParams.get("scalebar." + attr, default)
        return value

    rotation = _get_value("rotation", "horizontal").lower()
    length_fraction = _get_value("length_fraction", 0.2)
    fixed_value = scalebar.fixed_value
    fixed_units = scalebar.fixed_units or scalebar.units

    if rotation == "vertical":
        xlim, ylim = ylim, xlim

    if scalebar.fixed_value is None:
        length_px = abs(xlim[1] - xlim[0]) * length_fraction
        length_px, value, units = scalebar._calculate_best_length(length_px)
    else:
        value = fixed_value
        units = fixed_units

    return scalebar.scale_formatter(value, scalebar.dimension.to_latex(units))


class DataSequence(ProcessingResult, ABC):
    """
    Abstract base class for defining a sequence of data, e.g. raw data, reconstructed images, unmixed images. Enables
    consistent saving and processing for all of these data types.
    """
    n_im_dim = 3

    @property
    def attributes(self):
        return self.da.attrs

    def __init__(self, data, dimensions, coordinates=None, attributes=None, hdf5_sub_name=None,
                 algorithm_id=None):
        if coordinates is None:
            coordinates = {x: np.arange(data.shape[i]) for i, x in enumerate(dimensions)}
        ProcessingResult.__init__(self)
        self.da = DataArray(data=data, dims=dimensions, coords=coordinates,
                            attrs=attributes)
        self._cmap = None
        self.hdf5_sub_name = hdf5_sub_name
        self.algorithm_id = algorithm_id

    def __getitem__(self, item):
        c = self.copy()
        try:
            c.da = c.da[item]
        except TypeError:
            # Fudging the JAX implementation...
            c.da = c.da.copy()
            c.da.variable._data = np.asarray(c.da.variable._data)
            c.da = c.da[item]
        return c

    # TODO: Implement a concatenate function
    def copy(self, cls=None):
        from copy import copy
        c = copy(self)
        if cls is not None:
            c.__class__ = cls
        return c

    @property
    def shape_2d(self):
        if np.any(np.array(self.raw_data.shape[-3:]) == 1):
            return tuple([x for x in self.raw_data.shape[-3:] if x != 1])
        else:
            return self.raw_data.shape[-2:]

    @property
    def extent(self):
        coords = [self.da.coords[x] for x in self.two_dims()]
        return sum(((np.min(np.array(c)), np.max(np.array(c))) for c in coords), ())

    def two_dims(self):
        if self.da.coords["x"].size == 1:
            return "y", "z"
        elif self.da.coords["y"].size == 1:
            return "x", "z"
        else:
            return "x", "y"

    def to_2d(self):
        s = (0, ) * (len(self.shape) - self.n_im_dim)
        if self.n_im_dim > 2:
            slicer = [0] * self.n_im_dim
            for i in np.argsort(self.shape[-self.n_im_dim:])[-2:]:
                slicer[i] = slice(None)
            s += tuple(slicer)
        return self[s]

    def imshow(self, ax=None, roi_mask: Tuple["ROI", Iterable["ROI"]] = None,
               mask_roi=True,
               cmap=None, scale_kwargs=None, return_scalebar_dimension=False, scalebar=True,
               **kwargs):
        # TODO: make roi_mask take an array
        if scale_kwargs is None:
            scale_kwargs = {}

        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        if roi_mask is not None:
            if type(roi_mask) is not ROI:
                try:
                    mask, image_slice = roi_mask[0].to_mask_slice(self)
                    image_slice = image_slice.to_2d()
                    display_image = np.squeeze(image_slice.numpy_array).astype(np.float)
                    overall_mask = np.zeros(display_image.shape, dtype=np.bool)
                    for roi in roi_mask:
                        mask, _ = roi.to_mask_slice(self)
                        if mask_roi:
                            overall_mask[np.squeeze(mask)] = True
                    if mask_roi:
                        display_image[~overall_mask] = np.nan
                except TypeError:
                    raise ValueError("roi_mask must be a ROI or a tuple of ROIs")
            else:
                mask, image_slice = roi_mask.to_mask_slice(self)
                image_slice = image_slice.to_2d()
                display_image = np.squeeze(image_slice.numpy_array).astype(np.float)
                if mask_roi:
                    display_image[~np.squeeze(mask)] = np.nan
        else:
            display_image = self.to_2d().numpy_array

        interpolation = "nearest"
        if display_image.dtype == np.bool_:
            interpolation = "nearest"

        if cmap is None:
            cmap = self.cmap
        if np.iscomplexobj(display_image):
            display_image = np.real(display_image)
        display_image = np.squeeze(display_image)
        if "origin" not in kwargs:
            kwargs["origin"] = "lower"
        im = ax.imshow(display_image, extent=self.extent,
                       cmap=cmap, **kwargs, interpolation=interpolation)
        ax.axis("off")
        if scalebar:
            from matplotlib_scalebar.scalebar import ScaleBar
            scale_kwargs_defaults = dict(length_fraction=0.1, location="lower right",
                                         font_properties=dict(size="xx-small"), box_alpha=0., color="w")
            scale_kwargs_defaults.update(scale_kwargs)
            scalebar = ScaleBar(1, "m", **scale_kwargs_defaults)
            ax.add_artist(scalebar)
        if return_scalebar_dimension:
            return im, _get_matplotlib_scalebar_size(scalebar)
        else:
            return im

    @property
    def cmap(self):
        if self._cmap is None:
            return type_cmaps[self.get_hdf5_group_name()]
        else:
            return self._cmap

    @cmap.setter
    def cmap(self, x):
        self._cmap = x

    @staticmethod
    def get_ax1_label_meaning():
        return ""

    @abstractmethod
    def get_hdf5_group_name(self):
        pass

    @property
    def values(self):
        return self.raw_data

    @property
    def numpy_array(self):
        return np.asarray(self.values)

    @property
    def raw_data(self):
        # TODO: check if this has side effects in future. Previously, this said if type(self.da.data) == Array:
        if type(self.da.variable._data) == DaskArray:
            return np.array(self.da.data)
        else:
            return self.da.variable._data

    @raw_data.setter
    def raw_data(self, value):
        self.da.values = value

    @property
    def shape(self):
        return self.da.shape

    @property
    def ndim(self):
        return self.da.ndim

    @property
    def dtype(self):
        return self.da.dtype

    @property
    def ax_1_labels(self):
        return self.da.coords.get(self.get_ax1_label_meaning(), None)

    @property
    def ax_0_labels(self):
        return self.da.coords["frames"]

    @staticmethod
    def ax_0_exists():
        return True


class ImageSequence(DataSequence):
    @staticmethod
    def is_single_instance():
        return False

    def get_hdf5_group_name(self):
        raise NotImplementedError()

    def __add__(self, other):
        # A really lazy implementation of concatenating these datasets. There is 100% a better way to do this..
        new_data = xarray.concat([self.da, other.da], dim=other.da.dims[0])
        output = ImageSequence(new_data.values, self.ax_1_labels, self.algorithm_id, self.fov_3d, self.attributes,
                               self.hdf5_sub_name, ax1_meaning=self.get_ax1_label_meaning())
        output.__class__ = self.__class__
        return output

    @property
    def fov(self):
        n_pixel_tags = [ReconAttributeTags.X_NUMBER_OF_PIXELS,
                        ReconAttributeTags.Y_NUMBER_OF_PIXELS,
                        ReconAttributeTags.Z_NUMBER_OF_PIXELS]
        fov_tags = [ReconAttributeTags.X_FIELD_OF_VIEW,
                    ReconAttributeTags.Y_FIELD_OF_VIEW,
                    ReconAttributeTags.Z_FIELD_OF_VIEW]
        n_pixels = np.array([self.attributes.get(tag, 1) for tag in n_pixel_tags])
        if np.all(n_pixels == 1):
            # Old-style data
            n_pixels = np.array([self.attributes.get(ReconAttributeTags.OLD_RECON_NX)] * 2)
            fov_tags = [ReconAttributeTags.OLD_FIELD_OF_VIEW] * 2
        axes = np.where(~(n_pixels == 1))[0]
        fov_x = self.attributes.get(fov_tags[axes[0]], None)
        fov_y = self.attributes.get(fov_tags[axes[1]], None)
        return fov_x, fov_y

    @property
    def fov_3d(self):
        fov_tags = [ReconAttributeTags.X_FIELD_OF_VIEW,
                    ReconAttributeTags.Y_FIELD_OF_VIEW,
                    ReconAttributeTags.Z_FIELD_OF_VIEW]
        fov = np.array([self.attributes.get(tag, None) for tag in fov_tags])
        if all([x is None for x in fov_tags]):
            # Old-style data
            n_pixels = self.raw_data.shape[-3:]
            fov = np.array(
                [self.attributes.get(ReconAttributeTags.OLD_FIELD_OF_VIEW) if x != 1 else 1 for x in n_pixels])
        return fov

    def __init__(self, raw_data, ax_1_labels=None,
                 algorithm_id="", field_of_view=None,
                 attributes=None, hdf5_sub_name=None, ax1_meaning=None):
        # Ax1 labels = e.g. Wavelength
        if ax1_meaning is None:
            ax1_meaning = self.get_ax1_label_meaning()

        # Quick bit of validation
        if ax_1_labels is not None and ax1_meaning is not None:
            if raw_data.shape[1] != len(ax_1_labels):
                raise ValueError("Axis 1 labels must match raw data size.")

        if type(field_of_view[0]) is not tuple:
            field_of_view = [(-x / 2, x / 2) if x is not None else (0, 0) for x in field_of_view]

        xs = [np.linspace(x, y, N) for (x, y), N in zip(field_of_view, raw_data.shape[-3:][::-1])]

        dims = ["frames", "z", "y", "x"]
        coords = {"frames": np.arange(raw_data.shape[0]),
                  "x": xs[0],
                  "y": xs[1],
                  "z": xs[2]
                  }

        if not self.ax_0_exists():
            del coords["frames"]
            dims = dims[1:]

        if ax1_meaning is not None:
            dims.insert(1, ax1_meaning)
            coords[ax1_meaning] = ax_1_labels
        else:
            # If there isn't really an axis 1 (e.g. for delta so2).
            coords[ax1_meaning] = ax_1_labels[0]

        DataSequence.__init__(self, raw_data, dims, coords, attributes,
                              hdf5_sub_name, algorithm_id)
