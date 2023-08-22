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
            kwargs["model_geometry"] = kwargs.get("model_geometry", pa_example.get_scan_geometry())
            kwargs["model_fs"] = kwargs.get("model_fs", pa_example.get_sampling_frequency())
            kwargs["model_nt"] = kwargs.get("model_nt", pa_example.get_time_series().shape[-1])
            kwargs["model_c"] = kwargs.get("model_c", pa_example.get_speed_of_sound())
            kwargs["model_irf"] = kwargs.get("model_irf", pa_example.get_impulse_response())
        super().__init__(n_pixels, field_of_view, **kwargs)
        self._batch = False
        axes = [i for i, x in enumerate(n_pixels) if x != 1]
        dx = field_of_view[axes[0]] / (n_pixels[axes[0]] - 1)
        if dx != field_of_view[axes[1]] / (n_pixels[axes[1]] - 1) or field_of_view[axes[1]] != field_of_view[axes[0]]:
            raise ValueError("Current model based implementation only supports square reconstruction grids.")

        detx = kwargs["model_geometry"][:, axes[0]]
        dety = kwargs["model_geometry"][:, axes[1]]
        fs = kwargs["model_fs"]
        nx = n_pixels[axes[0]]
        x_0 = - field_of_view[axes[0]] / 2
        nt = kwargs["model_nt"]
        c = kwargs["model_c"]
        self._nx_model = nx
        self._model_matrix = self._generate_model(detx, dety, fs, dx, nx, x_0,
                                                  nt, c)
        self._model_matrix.eliminate_zeros()
        self._model_matrix.sort_indices()
        from jax.experimental.sparse import CSR

        self._model_regulariser = (kwargs.get("model_regulariser", None), kwargs.get("model_regulariser_lambda", None))
        if self._model_regulariser[0] is None:
            self._model_regulariser = None

        self._model_matrix = CSR((self._model_matrix.data, self._model_matrix.indices, self._model_matrix.indptr),
                                 shape=self._model_matrix.shape)

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
                    n_pixels=None,
                    field_of_view=None,
                    speed_of_sound=None,
                    **kwargs) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        from jax.scipy.signal import convolve2d
        import jaxopt
        from jaxopt.projection import projection_non_negative
        from tqdm.auto import tqdm
        # from jax.scipy.signal import convolve2d
        M = self._model_matrix

        @jax.jit
        def forward(params, y, M, regulariser=None):
            x = M @ params.flatten()
            residuals = x - y.flatten()
            loss1 = jnp.mean(residuals ** 2)
            if regulariser is not None:
                method, lambda_reg = regulariser
                if method == "identity":
                    loss2 = lambda_reg * jnp.mean(params ** 2)
                elif method == "laplacian":
                    conv_mat = jnp.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8
                    loss2 = lambda_reg * jnp.mean(convolve2d(params, conv_mat) ** 2)
                else:
                    loss2 = 0
            else:
                loss2 = 0
            return loss1 + loss2, {"loss1": loss1, "loss2": loss2}  # value, aux

        def rec(time_series):
            opt = jaxopt.ProjectedGradient(forward, projection=projection_non_negative,
                                           maxiter=50, has_aux=True, acceleration=True)
            result = opt.run(jnp.zeros((self._nx_model, self._nx_model)),
                             y=jnp.array(time_series), M=M, regulariser=self._model_regulariser)
            return result.params

        output_shape = raw_data.shape[:-2]
        raw_data = raw_data.reshape((-1,) + raw_data.shape[-2:])
        output = np.zeros((raw_data.shape[0], M.shape[1]))
        for i in tqdm(range(raw_data.shape[0])):
            ts = jnp.array(raw_data[i] - np.mean(raw_data[i], axis=-1)[:, None])
            output[i] = np.array(rec(raw_data[i]).reshape(output[i].shape))
        return output.reshape(output_shape + self.n_pixels)

    @staticmethod
    def get_algorithm_name() -> str:
        return "JAX Model Based Reconstruction"
