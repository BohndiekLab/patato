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
        Create ROI shape instances based on type.
        Type is specified in the "__classname" key
        """

        class_name = roi_data.get("__classname")
        if class_name == "iROI":
            # if type not specified, ilabs means it is a rectangle
            roi_type = roi_data.get("type", "Rectangle")
            if roi_type == "Ellipse":
                return EllipseROI(roi_data)
            elif roi_type == "Segment":
                # segment (line) can be treated as a polygon with 2 points
                return PolygonROI(roi_data)
            elif roi_type == "Polygon":
                return PolygonROI(roi_data)
            elif roi_type == "Rectangle":
                return RectangleROI(roi_data)
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

    @staticmethod
    def extract_roi_coords(roi_string):
        """
        Extract the (x, y, z) coordinates from the pos or size string.
        """
        roi_coords = re.findall(r"\((.*?)\)", roi_string)[0].split(", ")
        return [float(c) for c in roi_coords]


class EllipseROI(IROIShape):
    """
    Ellipse ROI shape.
    Given as pos and size in the iannotation file.
    pos refers to the top-left corner of the enclosing rectangle of the ellipse.
    Rotation angle (in degrees) is given relative to top-left corner.
    size refers to the total width and height of the ellipse.
    top-left corner is the origin of the coordinate system.
    """

    def __init__(self, roi_data):

        # Size not given means circle with d = 0.01m in ilabs
        self._size = (
            [0.01, 0.01]
            if "size" not in roi_data
            else self.extract_roi_coords(roi_data["size"])
        )

        self._topleft = self.extract_roi_coords(roi_data["pos"])

        # Calculate unrotated center (assuming the rectangle is axis-aligned)
        unrotated_center = [
            self._topleft[0] + self._size[0] / 2,
            self._topleft[1] + self._size[1] / 2,
        ]

        # if ellipse was rotated, angle is given
        self._angle = 0 if "angle" not in roi_data else roi_data["angle"]

        if self._angle == 0:
            # no rotation
            self._center = unrotated_center
        else:
            # Calculate the rotated center
            angle_rad = np.radians(self._angle)

            # Rotate the center around the top-left corner by the angle
            self._center = [
                self._topleft[0]
                + (unrotated_center[0] - self._topleft[0]) * np.cos(angle_rad)
                - (unrotated_center[1] - self._topleft[1]) * np.sin(angle_rad),
                self._topleft[1]
                + (unrotated_center[0] - self._topleft[0]) * np.sin(angle_rad)
                + (unrotated_center[1] - self._topleft[1]) * np.cos(angle_rad),
            ]

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

    @property
    def angle(self):
        return self._angle

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
        Includes rotation of the ellipse based on the specified angle
        """
        h, k = self._center[:2]
        a = self._size[0] / 2  # Semi-major axis
        b = self._size[1] / 2  # Semi-minor axis
        angle_rad = np.deg2rad(self._angle)  # Rotation angle in radians

        # Generate theta values from 0 to 2π for n_points evenly spaced points
        theta = np.linspace(0, 2 * np.pi, n_points)

        # Parameterized coordinates of the ellipse before rotation
        x = a * np.cos(theta)
        y = b * np.sin(theta)

        if self._angle != 0:
            # Apply rotation matrix for the specified angle and shift the ellipse points to center
            x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

            x_final = h + x_rot
            y_final = k + y_rot
        else:
            # simply shift the ellipse points to center
            x_final = h + x
            y_final = k + y

        return np.vstack((x_final, y_final)).T

    def __str__(self):
        return f"Ellipse with center at {self._center} and size {self._size} (Area: {self.area()})"


class PolygonROI(IROIShape):
    """
    Polygon ROI shape.
    Can also be used to represent a line segment.
    Given as a list of points in the iannotation file.
    top-left corner is the origin of the coordinate system.
    """

    def __init__(self, roi_data):
        points_dict = roi_data["points"]
        self._points = np.array([self.extract_roi_coords(p) for p in points_dict])

    @property
    def points(self):
        return self._points

    def area(self):
        """
        Area of the polygon using the shoelace formula.
        """

        if len(self._points) < 3:
            return 0

        x = self._points[:, 0]
        y = self._points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def discretize(self, n_points=100):
        """
        Polygon is already a set of points, so just return them.
        """
        return self._points

    def __str__(self):
        return f"Polygon with points {self._points} (Area: {self.area()})"


class RectangleROI(IROIShape):
    """
    Rectangle ROI shape.
    Given as pos and size in the iannotation file.
    pos refers to the top-left corner of the rectangle.
    top-left corner is the origin of the coordinate system.
    """

    def __init__(self, roi_data):
        self._topleft = self.extract_roi_coords(roi_data["pos"])

        # size not given means square with d = 0.01m in ilabs
        self._size = (
            [0.01, 0.01]
            if "size" not in roi_data
            else self.extract_roi_coords(roi_data["size"])
        )

    @property
    def topleft(self):
        return self._topleft

    @property
    def size(self):
        return self._size

    def area(self):
        """
        Area of the rectangle.
        """
        return self._size[0] * self._size[1]

    def discretize(self, n_points=4):
        """
        Return the four corners of the rectangle as a set of points.
        """
        x, y = self._topleft
        w, h = self._size
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    def __str__(self):
        return f"Rectangle with top-left corner at {self._topleft} and size {self._size} (Area: {self.area()})"


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
