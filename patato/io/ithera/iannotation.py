import json
import os
import re
import math
from abc import ABC, abstractmethod

import numpy as np


class IROIShape(ABC):
    """
    Base class for all ROI shapes.
    """

    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def discretize(self, n_points):
        """
        Discretize the shape into a set of points.
        """
        pass

    @staticmethod
    def create_roi(roi_data):
        """
        Factory method to create specific ROI shape instances based on type.
        Type is specified in the "__classname" key
        """

        class_name = roi_data.get("__classname")
        if class_name == "iROI":
            roi_type = roi_data.get("type")
            if roi_type == "Ellipse":
                return EllipseROI(roi_data)
            else:
                raise NotImplementedError(f"ROI shape '{roi_type}' is not implemented.")
        else:
            raise NotImplementedError(
                f"annotation class '{class_name}' is not implemented."
            )

    @staticmethod
    def scale_points(coords, x_scale=1, y_scale=1):
        """
        Scale all x and y coordinates in the list of coordinates by the given x and y scale factors.
        coords: numpy array of shape (n_points, 2)
        """
        return np.array([[x * x_scale, y * y_scale] for x, y in coords])


class EllipseROI(IROIShape):
    """
    Ellipse ROI shape.
    Given as pos and size in the iannotation file.
    pos refers to the top-left corner of the enclosing rectangle of the ellipse.
    size refers to the total width and height of the ellipse.
    top-left corner is the origin of the coordinate system.
    """

    def __init__(self, roi_data):
        self.topleft = self.extract_roi_coords(roi_data["pos"])
        self._size = self.extract_roi_coords(roi_data["size"])
        self._center = [
            self.topleft[0] + self._size[0] / 2,
            self.topleft[1] + self._size[1] / 2,
        ]

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

    @staticmethod
    def extract_roi_coords(roi_string) -> list:
        """
        Extract the (x, y, z) coordinates from the pos or size string.
        """
        roi_coords = re.findall(r"\((.*?)\)", roi_string)[0].split(", ")
        return [float(c) for c in roi_coords]

    def area(self):
        """
        Area of the ellipse.
        (π * semi-major * semi-minor)
        """
        semi_major = self._size[0] / 2
        semi_minor = self._size[1] / 2
        return math.pi * semi_major * semi_minor

    def discretize(self, n_points=100):
        """
        Discretize the ellipse boundary into n_points, returning them as a numpy array of shape (n_points, 2).
        This is not an ideal way for storing ellipses, but ensures easy compatibility with patatos existing rois.
        """
        h, k = self._center[:2]  # Center coordinates (ignoring z for 2D representation)
        a = self._size[0] / 2  # Semi-major axis (half of the width)
        b = self._size[1] / 2  # Semi-minor axis (half of the height)

        theta = np.linspace(0, 2 * np.pi, n_points)  # Angles evenly spaced from 0 to 2π
        x = h + a * np.cos(theta)  # X-coordinates of the ellipse points
        y = k + b * np.sin(theta)  # Y-coordinates of the ellipse points

        return np.vstack(
            (x, y)
        ).T  # Stack x and y into a 2D array of shape (n_points, 2)

    def __str__(self):
        return f"Ellipse with center at {self._center} and size {self._size} (Area: {self.area()})"


class IAnnotation:
    """
    Class to load and parse iThera iannotation files.
    """

    def _load_iannotation_file(self):
        """
        Load the iannotation file and parse the data.
        """

        with open(self.path, "r") as json_file:
            data = json.load(json_file)

        self.scan_hash = data.get("ScanHash", None)
        annotations = []

        for annotation in data["Annotations"]:
            roi_list = []
            for roi in annotation["ROIList"]:
                roi_object = IROIShape.create_roi(roi)
                roi_list.append(roi_object)

            classname = annotation.get("__classname", None)
            sweeps = annotation.get("Sweeps", [])
            source = annotation.get("Source", None)

            annotations.append(
                {
                    "ROIs": roi_list,
                    "Classname": classname,
                    "Sweeps": sweeps,
                    "Source": source,
                }
            )

        self.annotations = annotations
        self.n_annotations = len(annotations)

    def __init__(self, file_path, unit="m"):

        self.unit = unit

        if os.path.exists(file_path):
            self.path = file_path
            self._load_iannotation_file()


# # Example Usage
# file_path = (
#     "/Projects/PATATO-Annotations/data/Scan_11.iannotation"
# )
# iannotation_object = IAnnotation(file_path)

# for annotation in iannotation_object.annotations:

#     print(f"Classname: {annotation['Classname']}")
#     print(f"Sweeps: {annotation['Sweeps']}")
#     print(f"Source: {annotation['Source']}")

#     for roi in annotation["ROIs"]:
#         print(roi)
#         print(roi.discretize(10))
