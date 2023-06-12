#  Copyright (c) Thomas Else 2023.
#  License: MIT

from typing import Sequence

from .gpu_preprocessing_algorithm import GPUMSOTPreProcessor
from .preprocessing_algorithm import NumpyPreProcessor
from .jax_preprocessing_algorithm import PreProcessor
from .processing_algorithm import TimeSeriesProcessingAlgorithm


PREPROCESSING_METHODS: Sequence[type(TimeSeriesProcessingAlgorithm)] = [NumpyPreProcessor,
                                                                        GPUMSOTPreProcessor,
                                                                        PreProcessor]

PREPROCESSING_NAMES = {x.get_algorithm_name(): x for x in PREPROCESSING_METHODS}
