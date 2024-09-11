#  Copyright (c) Thomas Else 2023.
#  License: MIT
from typing import Sequence

import numpy as np

from .. import ReconstructionAlgorithm
import numpy.typing as npt
from ... import PAData
from ...processing.preprocessing_algorithm import (
    TimeSeriesProcessingAlgorithm,
    PreprocessingAttributeTags,
)


class ModelBasedPreProcessor(TimeSeriesProcessingAlgorithm):
    @staticmethod
    def get_algorithm_name() -> str:
        """
        Get the name of the algorithm.

        Returns
        -------
        str or None
        """
        return "Model Based Preprocessor"

    @staticmethod
    def get_hdf5_group_name():
        """
        Return the name of the group in the HDF5 file.

        Returns
        -------
        str or None
        """
        return None

    def __init__(self):
        super().__init__()

    def run(self, time_series, pa_data, **kwargs):
        new_ts = time_series.copy()
        ts_raw = (
            time_series.raw_data - np.mean(time_series.raw_data, axis=-1)[:, :, :, None]
        )

        if pa_data is not None:
            overall_correction_factor = pa_data.get_overall_correction_factor()
            ts_raw /= overall_correction_factor[:, :, None, None]
        else:
            overall_correction_factor = None
        new_ts.raw_data = ts_raw

        # Update the results' attributes.
        for a in time_series.attributes:
            if a not in new_ts.attributes:
                new_ts.attributes[a] = time_series.attributes[a]
        new_ts.attributes[
            PreprocessingAttributeTags.PROCESSING_ALGORITHM
        ] = self.get_algorithm_name()
        new_ts.attributes["CorrectionFactorApplied"] = (
            overall_correction_factor is not None
        )

        return new_ts, {}, None


class JAXModelBasedReconstruction(ReconstructionAlgorithm):
    """JAX-based, two-dimensional model based reconstruction algorithm."""

    def __init__(self, n_pixels, field_of_view, pa_example: "PAData" = None, **kwargs):
        if pa_example is not None:
            kwargs["model_geometry"] = kwargs.get(
                "model_geometry", pa_example.get_scan_geometry()
            )
            kwargs["model_fs"] = kwargs.get(
                "model_fs", pa_example.get_sampling_frequency()
            )
            kwargs["model_nt"] = kwargs.get(
                "model_nt", pa_example.get_time_series().shape[-1]
            )
            kwargs["model_c"] = kwargs.get("model_c", pa_example.get_speed_of_sound())
            kwargs["model_irf"] = kwargs.get(
                "model_irf", pa_example.get_impulse_response()
            )
        super().__init__(n_pixels, field_of_view, **kwargs)
        self._batch = False
        axes = [i for i, x in enumerate(n_pixels) if x != 1]
        dx = field_of_view[axes[0]] / (n_pixels[axes[0]] - 1)
        if (
            dx != field_of_view[axes[1]] / (n_pixels[axes[1]] - 1)
            or field_of_view[axes[1]] != field_of_view[axes[0]]
        ):
            raise ValueError(
                "Current model based implementation only supports square reconstruction grids."
            )

        detx = kwargs["model_geometry"][:, axes[0]]
        dety = kwargs["model_geometry"][:, axes[1]]
        fs = kwargs["model_fs"]
        nx = n_pixels[axes[0]]
        x_0 = -field_of_view[axes[0]] / 2
        nt = kwargs["model_nt"]
        c = kwargs["model_c"]
        self._nx_model = nx
        self._model_matrix = self._generate_model(detx, dety, fs, dx, nx, x_0, nt, c)
        self._model_matrix.eliminate_zeros()
        self._model_matrix.sort_indices()
        from jax.experimental.sparse import CSR

        self._model_regulariser = (
            kwargs.get("model_regulariser", None),
            kwargs.get("model_regulariser_lambda", None),
        )
        # if self._model_regulariser[0] is None:
        #    self._model_regulariser = None

        self._model_matrix = CSR(
            (
                self._model_matrix.data,
                self._model_matrix.indices,
                self._model_matrix.indptr,
            ),
            shape=self._model_matrix.shape,
        )

        self._model_max_iter = kwargs.get("model_max_iter", 50)
        self._model_constraint = kwargs.get("constraint", "positive")

        self.attributes["model_max_iter"] = self._model_max_iter
        self.attributes["model_constraint"] = self._model_constraint
        self.attributes["model_regulariser"] = self._model_regulariser[0]
        self.attributes["model_regulariser_weighting"] = self._model_regulariser[1]
        self.attributes["model_c"] = c

    def _generate_model(
        self,
        detx: npt.ArrayLike,
        dety: npt.ArrayLike,
        fs: float,
        dx: float,
        nx: int,
        x_0: float,
        nt: int,
        c: float,
    ):
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

    def reconstruct(
        self,
        raw_data: np.ndarray,
        fs: float = None,
        geometry: np.ndarray = None,
        n_pixels=None,
        field_of_view=None,
        speed_of_sound=None,
        **kwargs
    ) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        from jax.scipy.signal import convolve2d
        import jaxopt
        from jaxopt.projection import projection_non_negative, projection_box
        from tqdm.auto import tqdm

        if self._model_constraint == "positive":
            projection = projection_non_negative
            hyperparams = None
        elif self._model_constraint == "none":
            projection = projection_box
            hyperparams = (-np.inf, np.inf)
        else:
            raise ValueError("Constraint must either be 'positive' or 'none'.")

        M = self._model_matrix
        if self._model_regulariser is not None or all(
            [x is None for x in self._model_regulariser]
        ):
            method, lambda_reg = self._model_regulariser
        else:
            method, lambda_reg = None, None

        if method == "laplacian":
            conv_mat = jnp.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8
        else:
            conv_mat = None

        if method not in ["identity", "laplacian", None]:
            raise ValueError(
                "Regularisation method must either be identity, laplacian or None."
            )

        @jax.jit
        def forward(params, y, M, lambda_reg=None, conv_mat=None):
            x = M @ params.flatten()
            residuals = x - y.flatten()
            loss1 = jnp.mean(residuals**2)
            if lambda_reg is not None:
                if conv_mat is None:
                    loss2 = lambda_reg * jnp.mean(params**2)
                else:
                    loss2 = lambda_reg * jnp.mean(convolve2d(params, conv_mat) ** 2)
            else:
                loss2 = 0
            return loss1 + loss2, {"loss1": loss1, "loss2": loss2}  # value, aux

        def rec(time_series):
            opt = jaxopt.ProjectedGradient(
                forward,
                projection=projection,
                maxiter=self._model_max_iter,
                has_aux=True,
                acceleration=True,
            )
            result = opt.run(
                jnp.zeros((self._nx_model, self._nx_model)),
                y=jnp.array(time_series),
                M=M,
                lambda_reg=lambda_reg,
                conv_mat=conv_mat,
                hyperparams_proj=hyperparams,
            )
            return result.params, result.state

        output_shape = raw_data.shape[:-2]
        raw_data = raw_data.reshape((-1,) + raw_data.shape[-2:])
        output = np.zeros((raw_data.shape[0], M.shape[1]))
        for i in tqdm(range(raw_data.shape[0]), leave=False):
            ts = jnp.array(raw_data[i] - np.mean(raw_data[i], axis=-1)[:, None])
            params, state = rec(ts)
            output[i] = np.array(params.reshape(output[i].shape))
            self._prev_state = state
        return output.reshape(output_shape + self.n_pixels[::-1])

    @staticmethod
    def get_algorithm_name() -> str:
        return "JAX Model Based Reconstruction"
