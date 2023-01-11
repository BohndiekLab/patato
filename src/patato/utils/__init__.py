#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

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
