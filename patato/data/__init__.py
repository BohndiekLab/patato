#  Copyright (c) Thomas Else 2023.
#  License: MIT

"""
PATATO data examples
====================

PATATO data - an interface for exemplar data and basic simulations to illustrate the core features of PATATO.
"""

from .simulated_datasets import get_basic_p0, generate_basic_simulation
from .get_example_datasets import get_msot_time_series_example, get_ithera_msot_time_series_example, \
    get_msot_phantom_example
