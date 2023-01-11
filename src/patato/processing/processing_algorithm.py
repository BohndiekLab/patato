#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
from typing import Union, List, Any, Tuple

if TYPE_CHECKING:
    from ..io.msot_data import PAData
    from ..core.image_structures.image_sequence import ImageSequence
    from ..core.image_structures.pa_time_data import PATimeSeries


class ProcessingResult(ABC):
    save_output = False

    @abstractmethod
    def __add__(self, other):
        pass

    @property
    def attributes(self):
        return {}

    def __init__(self):
        self.algorithm_id = ""
        self.hdf5_sub_name = ""

    @property
    @abstractmethod
    def raw_data(self):
        pass

    @abstractmethod
    def get_hdf5_group_name(self):
        pass

    @staticmethod
    @abstractmethod
    def is_single_instance():
        pass

    def get_hdf5_sub_name(self):
        return self.hdf5_sub_name

    def save(self, scan_writer):
        scan_writer.add_image(self)


class ProcessingAlgorithm(ABC):
    def __init__(self):
        self.children: List["ProcessingAlgorithm"] = []

    @abstractmethod
    def run(self, input_data: Any,
            pa_data: PAData, **kwargs) -> Optional[Tuple[ProcessingResult, dict, Optional[List[ProcessingResult]]]]:
        pass

    def add_child(self, child: "ProcessingAlgorithm"):
        self.children.append(child)


class TimeSeriesProcessingAlgorithm(ProcessingAlgorithm, ABC):
    def __init__(self):
        ProcessingAlgorithm.__init__(self)

    @abstractmethod
    def run(self, time_series: PATimeSeries,
            pa_data: PAData, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def get_algorithm_name() -> Union[str, None]:
        pass


class SpatialProcessingAlgorithm(ProcessingAlgorithm, ABC):
    def __init__(self, algorithm_id=""):
        ProcessingAlgorithm.__init__(self)
        self.algorithm_id = algorithm_id

    @abstractmethod
    def run(self, spatial_data: ImageSequence,
            pa_data: PAData, **kwargs):
        pass

    def get_algorithm_id(self) -> Union[str, None]:
        return self.algorithm_id
