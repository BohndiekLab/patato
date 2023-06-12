#  Copyright (c) Thomas Else 2023.
#  License: MIT

"""
Utility module
=====================

This module provides utility functions for PATATO that are used in several other modules.
"""

from os.path import split
from typing import Tuple


def sort_key(file: str) -> Tuple[int, str]:
    """

    Parameters
    ----------
    file

    Returns
    -------

    """
    try:
        return int(split(file)[-1].split(".")[0].split("_")[-1]), file
    except ValueError:
        return 0, file
