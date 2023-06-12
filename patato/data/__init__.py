#  Copyright (c) Thomas Else 2023.
#  License: MIT

"""
Data examples
====================

PATATO data - an interface for exemplar data and basic simulations to illustrate the core features of PATATO.
"""

from .simulated_datasets import get_basic_p0, generate_basic_simulation
from .get_example_datasets import get_msot_time_series_example, get_ithera_msot_time_series_example, \
    get_msot_phantom_example

__all_exports = [get_msot_time_series_example, get_ithera_msot_time_series_example,
                 get_msot_phantom_example, get_basic_p0, generate_basic_simulation]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]
