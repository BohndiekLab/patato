from __future__ import annotations

from abc import ABCMeta, abstractmethod

from numpy.typing import ArrayLike


class DataInterface(metaclass=ABCMeta):
    """The abstract base class for interfacing between PATATO and datasets."""

    def __init__(self, shape: tuple[int]):
        """Initialise the DataInterface class.

        Parameters
        ----------
        shape : tuple
            The shape of the data interface (frames/wavelengths etc.). This should not include the detector shape or reconstructed image shape.
        """
        self._shape = shape

    def __getitem__(self, items: slice | tuple[int]) -> DataInterface:
        """The implementation of slicing for PATATO datasets.

        Parameters
        ----------
        items : slice
            The item selection for slicing.
        """
        return self

    @abstractmethod
    def _get_time_series(self) -> ArrayLike:
        """Get the full (unsliced) photoacoustic time series data for a given data interface (e.g. hdf5/numpy etc)."""

    def get_time_series(self) -> ArrayLike:
        """Get the photoacoustic time series data. Note that this dataset will be sliced if this class has been sliced."""
        return self._get_time_series()
