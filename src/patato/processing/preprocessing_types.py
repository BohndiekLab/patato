#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from typing import Sequence

from .gpu_preprocessing_algorithm import GPUMSOTPreProcessor
from .preprocessing_algorithm import DefaultMSOTPreProcessor
from .jax_preprocessing_algorithm import MSOTPreProcessor

# TODO: Change the type definition here (need to generalise the preprocessing method a bit better)

PREPROCESSING_METHODS: Sequence[type(DefaultMSOTPreProcessor)] = [DefaultMSOTPreProcessor, GPUMSOTPreProcessor,
                                                                  MSOTPreProcessor]

PREPROCESSING_NAMES = {x.get_algorithm_name(): x for x in PREPROCESSING_METHODS}
