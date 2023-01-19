#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from typing import Sequence

import numpy as np

from .backprojection_implementation.jax_implementation import full_recon
from .reconstruction_algorithm import ReconstructionAlgorithm

import jax


class ReferenceBackprojection(ReconstructionAlgorithm):
    """
    Reference backprojection: Uses JAX in the background.
    """

    def reconstruct(self, time_series: np.ndarray,
                    fs: float,
                    geometry: np.ndarray, n_pixels: Sequence[int],
                    field_of_view: Sequence[float],
                    speed_of_sound,
                    **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        time_series
        fs
        geometry
        n_pixels
        field_of_view
        speed_of_sound
        kwargs

        Returns
        -------

        """

        # Get parameters:
        dl = speed_of_sound / fs

        # Reshape frames so that we can loop through to reconstruct
        original_shape = time_series.shape[:-2]
        frames = int(np.product(original_shape))
        signal = time_series.reshape((frames,) + time_series.shape[-2:])

        dx = field_of_view[0] / (n_pixels[0] - 1) if n_pixels[0] != 1 else 0
        dy = field_of_view[1] / (n_pixels[1] - 1) if n_pixels[1] != 1 else 0
        dz = field_of_view[2] / (n_pixels[2] - 1) if n_pixels[2] != 1 else 0

        recon_all = jax.vmap(full_recon, in_axes=(0,) + (None,) * 8, out_axes=0)

        output = recon_all(signal, geometry, dl, n_pixels[0], n_pixels[1], n_pixels[2],
                           dx, dy, dz)

        return output.reshape(original_shape + tuple(n_pixels)[::-1])

    @staticmethod
    def get_algorithm_name() -> str:
        """

        Returns
        -------

        """
        return "Reference Backprojection"
