#  Copyright (c) Thomas Else 2023.
#  License: MIT

"""
PATATO Data
===========

PATATO Data - an interface for exemplar data and basic simulations.
"""

from .simulated_datasets import get_basic_p0, generate_basic_simulation
from .get_example_datasets import get_msot_time_series_example, get_ithera_msot_time_series_example, \
    get_msot_phantom_example
