#  Copyright (c) Thomas Else 2023.
#  License: MIT

from typing import Sequence

import numpy as np
from .. import ReconstructionAlgorithm

# Add a loading bar
from tqdm.auto import tqdm


class SlowBackprojection(ReconstructionAlgorithm):
    """
    Slow example backprojection.
    """

    def reconstruct(self, time_series: np.ndarray,
                    fs: float,
                    geometry: np.ndarray, n_pixels: Sequence[int],
                    field_of_view: Sequence[float],
                    speed_of_sound: float,
                    **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        time_series: array_like
            Photoacoustic time series data in a numpy array. Shape: (..., n_detectors, n_time_samples)
        fs: float
            Time series sampling frequency (Hz).
        geometry: array_like
            The detector geometry. Shape: (n_detectors, 3)
        n_pixels: tuple of int
            Tuple of length 3, (nx, ny, nz)
        field_of_view: tuple of float
            Tuple of length 3, (lx, ly, lz) - the size of the reconstruction volume.
        speed_of_sound: float
            Speed of sound (m/s).
        kwargs
            Extra parameters (optional), useful for advanced algorithms (e.g. multi speed of sound etc.).

        Returns
        -------
        array_like
            The reconstructed image.

        """
        print("Running batch of delay and sum reconstruction code.")

        # Get useful parameters:
        dl = speed_of_sound / fs

        # Reshape frames so that we can loop through to reconstruct
        original_shape = time_series.shape[:-2]
        frames = int(np.product(original_shape))
        signal = time_series.reshape((frames,) + time_series.shape[-2:])

        xs, ys, zs = [
            np.linspace(-field_of_view[i] / 2, field_of_view[i] / 2, n_pixels[i]) if n_pixels[i] != 1 else np.array(
                [0.]) for i in range(3)]
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')

        # Note that the reconstructions are stored in memory in the order z, y, x (i.e. the x axis is the fastest
        # changing in memory)
        output = np.zeros((frames,) + tuple(n_pixels)[::-1])

        for n_frame in tqdm(range(frames), desc="Looping through frames", position=0):
            for n_detector in tqdm(range(signal.shape[-2]), desc="Looping through detectors", position=1, leave=False):
                detx, dety, detz = geometry[n_detector]
                d = (np.sqrt((detx - X) ** 2 + (dety - Y) ** 2 + (detz - Z) ** 2) / dl).astype(np.int32)
                output[n_frame] += signal[n_frame, n_detector, d]
        return output.reshape(original_shape + tuple(n_pixels)[::-1])

    @staticmethod
    def get_algorithm_name() -> str:
        """

        Returns
        -------
        str
            Algorithm name.
        """
        return "Slow Backprojection"
