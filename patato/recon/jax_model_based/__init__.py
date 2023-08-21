#  Copyright (c) Thomas Else 2023.
#  License: MIT
from typing import Sequence

import numpy as np

from .. import ReconstructionAlgorithm
import numpy.typing as npt
from ... import PAData


class JAXModelBasedReconstruction(ReconstructionAlgorithm):
    """
    JAX-based, two-dimensional model based reconstruction algorithm.
    """

    def __init__(self, n_pixels, field_of_view, pa_example: "PAData" = None, **kwargs
                 ):
        if pa_example is not None:
            kwargs["geometry"] = kwargs.get("geometry", pa_example.get_scan_geometry())
            kwargs["fs"] = kwargs.get("fs", pa_example.get_sampling_frequency())
            kwargs["nt"] = kwargs.get("nt", pa_example.get_time_series().shape[-1])
            kwargs["c"] = kwargs.get("c", pa_example.get_speed_of_sound())
            kwargs["irf"] = kwargs.get("irf", pa_example.get_impulse_response())
        super().__init__(n_pixels, field_of_view, **kwargs)

        axes = [i for i, x in enumerate(n_pixels) if x != 1]
        dx = field_of_view[axes[0]] / (n_pixels[axes[0]] - 1)
        if dx != field_of_view[axes[1]] / (n_pixels[axes[1]] - 1) or field_of_view[axes[1]] != field_of_view[axes[0]]:
            raise ValueError("Current model based implementation only supports square reconstruction grids.")

        detx = kwargs["geometry"][:, axes[0]]
        dety = kwargs["geometry"][:, axes[1]]
        fs = kwargs["fs"]
        nx = n_pixels[axes[0]]
        x_0 = - field_of_view[axes[0]] / 2
        nt = kwargs["nt"]
        c = kwargs["c"]
        self._nx_model = nx
        self._model_matrix = self._generate_model(detx, dety, fs, dx, nx, x_0,
                                                  nt, c)

    def _generate_model(self, detx: npt.ArrayLike, dety: npt.ArrayLike,
                        fs: float, dx: float, nx: int, x_0: float, nt: int,
                        c: float):
        """
        Generates the model matrix for given parameters.

        Parameters
        ----------
        detx
        dety
        fs
        dx
        nx
        x_0
        nt

        Returns
        -------

        """
        from ..model_based.numpy_implementation import get_model
        dl = c / fs
        model_matrix = get_model(detx, dety, dl, dx, nx, x_0, nt, cache=False)
        return model_matrix

    def reconstruct(self, raw_data: np.ndarray,
                    fs: float = None,
                    geometry: np.ndarray = None,
                    n_pixels: Sequence[int] = None,
                    field_of_view: Sequence[float] = None,
                    speed_of_sound=None,
                    **kwargs) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        import jaxopt
        from jaxopt.projection import projection_non_negative
        # from jax.scipy.signal import convolve2d

        @jax.jit
        def forward(params, y):
            x = self._model_matrix @ params.flatten()
            residuals = x - y.flatten()
            loss1 = jnp.mean(residuals ** 2)
            # loss2 = l2 * jnp.mean(params**2)
            # loss2 = l2 * jnp.mean(convolve2d(params, conv_mat) ** 2)
            return loss1, {"loss1": loss1}  # value, aux

        @jax.jit
        def rec(time_series):
            opt = jaxopt.ProjectedGradient(forward, projection=projection_non_negative,
                                           maxiter=50, has_aux=True, acceleration=True)
            result = opt.run(jnp.zeros((self._nx_model, self._nx_model)),
                             y=time_series)
            return result.params

        return jax.vmap(rec, in_axes=(0, 1))(raw_data)

    @staticmethod
    def get_algorithm_name() -> str:
        return "JAX Model Based Reconstruction"
