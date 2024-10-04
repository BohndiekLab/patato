import json
from pathlib import Path
import re
import math
from abc import ABC, abstractmethod

import numpy as np


class IROIShape(ABC):
    """
    Base class for all ROI shapes.
    """

    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def discretize(self, n_points: int) -> np.ndarray:
        """
        Discretize the shape into a set of n points.
        """
        pass

    @property
    @abstractmethod
    def attributes(self) -> dict:
        pass

    @classmethod
    def create_roi(cls, roi_data: dict) -> "IROIShape":
        """
        Create ROI shape instances based on type.
        Type is specified in the "type" key for "iROI" class.
        """
        class_name = roi_data.get("__classname")
        if class_name == "iROI":
            roi_type = roi_data.get(
                "type", "Rectangle"
            )  # if type not specified, ilabs means it is a rectangle
            roi_classes = {
                "Ellipse": EllipseROI,
                "Segment": PolygonROI,  # segment is treated as a polygon with 2 points
                "Polygon": PolygonROI,
                "Rectangle": RectangleROI,
            }

            if roi_type in roi_classes:
                return roi_classes[roi_type](roi_data)
            else:
                raise NotImplementedError(f"ROI shape '{roi_type}' is not implemented.")
        else:
            raise NotImplementedError(
                f"Annotation class '{class_name}' is not implemented."
            )

    @staticmethod
    def close_polygon(points: np.ndarray) -> np.ndarray:
        """
        Close the polygon by adding the first point to the end.
        """
        return np.concatenate([points, points[0:1, :]])

    @staticmethod
    def scale_points(
        coords: np.ndarray, x_scale: float = 1.0, y_scale: float = 1.0
    ) -> np.ndarray:
        """
        Scale all x and y coordinates in the list of coordinates by the given x and y scale factors.
        coords: numpy array of shape (n_points, 2)
        """
        return np.array([[x * x_scale, y * y_scale] for x, y in coords])

    @staticmethod
    def extract_roi_coords(roi_string: str) -> list:
        """
        Extract the (x, y, z) coordinates from the pos or size string.
        """
        roi_coords = re.findall(r"\((.*?)\)", roi_string)[0].split(", ")
        return [float(c) for c in roi_coords]

    @property
    def type(self) -> str:
        return self._roi_type


class EllipseROI(IROIShape):
    """
    Ellipse ROI shape.
    Also used for circles.
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
    def type(self):
        return "Ellipse"

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

    @property
    def angle(self):
        return self._angle

    @property
    def attributes(self) -> dict:
        return {
            "Type": self.type,
            "Size": self._size,
            "Angle": self._angle,
            "TopLeft": self._topleft,
        }

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
    def type(self):
        return "Polygon"

    @property
    def points(self):
        return self._points

    @property
    def attributes(self) -> dict:
        return {
            "Type": self.type,
        }

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
    def type(self):
        return "Rectangle"

    @property
    def topleft(self):
        return self._topleft

    @property
    def size(self):
        return self._size

    @property
    def attributes(self) -> dict:
        return {
            "Type": self.type,
            "Size": self._size,
            "TopLeft": self._topleft,
        }

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

        with open(self._file_path, "r") as json_file:
            data = json.load(json_file)

        self._scan_hash = data.get("ScanHash", None)
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

        self._annotations = annotations
        self._n_annotations = len(annotations)

    def __init__(self, file_path, unit="m"):

        self.unit = unit
        file_path = Path(file_path)

        self._scan_hash = None
        self._annotations = []
        self._n_annotations = 0

        if file_path.exists():
            self._scan_name = file_path.parent.name
            self._study_name = file_path.parent.parent.name
            self._file_path = file_path
            self._load_iannotation_file()
        else:
            raise FileNotFoundError(f"iannotation file not found: {file_path}")

    @property
    def scan_name(self) -> str:
        return self._scan_name

    @property
    def study_name(self) -> str:
        return self._study_name

    @property
    def scan_hash(self) -> str:
        return self._scan_hash

    @property
    def annotations(self) -> list:
        return self._annotations

    def get_annotation_sweeps(self, anno_index: int) -> list:
        return self._annotations[anno_index]["Sweeps"]

    def get_annotation_source(self, anno_index: int) -> str:
        return self._annotations[anno_index]["Source"]

    def get_annotation_roi(self, anno_index: int, roi_index: int) -> IROIShape:
        return self._annotations[anno_index]["ROIs"][roi_index]

    def __str__(self):
        return f"iAnnotation object for {self.scan_name} with {self.n_annotations} annotations."

    def __len__(self):
        return self._n_annotations

    def __getitem__(self, anno_index):
        return self._annotations[anno_index]

    def __iter__(self):
        return iter(self.annotations)
