#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from __future__ import annotations

from typing import TypeVar

from .reconstruction_image import Reconstruction
from .single_image import SingleImage
from .single_parameter_data import SingleParameterData
from .unmixed_image import UnmixedData

T = TypeVar("T", bound="ImageSequence")
S = TypeVar("S", bound="ImageSequence")
IMAGE_DATA_TYPES = {"recons": Reconstruction, "unmixed": UnmixedData,
                    "so2": SingleParameterData, "thb": SingleParameterData,
                    "dso2": SingleImage, "dicg": SingleImage,
                    "baseline_so2_sigma": SingleImage, "baseline_so2": SingleImage,
                    "baseline_icg_sigma": SingleImage, "baseline_icg": SingleImage}
