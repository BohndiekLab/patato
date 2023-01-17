#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from typing import Sequence

import numpy as np
import pylops
from pylops.optimization.leastsquares import regularized_inversion
from pylops.signalprocessing import Convolve1D, Convolve2D
from scipy.sparse.linalg import LinearOperator as CPULinOp

from .cuda_implementation import get_model as get_model_gpu_single_c
from .cuda_implementation_refraction import get_model as get_model_gpu_double_c
from .numpy_implementation import generate_model as get_model_cpu_single_c
from .. import ReconstructionAlgorithm
from ...core.image_structures.pa_time_data import PATimeSeries

try:
    cuda_enabled = True
    from cupyx.scipy.sparse.linalg import LinearOperator as GPULinOp
    import cupy as cp
except:
    cuda_enabled = False


class ModelBasedReconstruction(ReconstructionAlgorithm):
    """Model based reconstruction algorithm processor.
    """
    def generate_model(self, detx, dety, fs, dx, nx, x_0, nt, gpu=True, cache=False, **kwargs):
        """

        Parameters
        ----------
        detx
        dety
        fs
        dx
        nx
        x_0
        nt
        gpu
        cache
        kwargs

        Returns
        -------

        """
        if gpu:
            get_model_single_c = get_model_gpu_single_c
            get_model_double_c = get_model_gpu_double_c
            LinearOperator = GPULinOp
        else:
            get_model_double_c = get_model_cpu_single_c
            get_model_single_c = get_model_gpu_single_c
            LinearOperator = CPULinOp
        model_type = self.custom_params.get("model_type", "single_sos")
        regulariser = self.custom_params.get("regulariser", self.custom_params.get("regularizer", None))
        if regulariser not in ["identity", "laplacian", "TV"]:
            raise ValueError("Invalid regulariser.")
        reg_lambda = self.custom_params.get("reg_lambda", 1)
        iter_lim = self.custom_params.get("iter_lim", 10)

        # generate a model.
        if model_type == "single_sos":
            c = kwargs.get("c", None) or self.custom_params.get("c")
            dl = c / fs
            model_matrix = get_model_single_c(detx, dety, dl, dx, nx,
                                              x_0, nt, cache=cache)
        elif model_type == "refraction":
            c0 = kwargs.get("c0", None) or self.custom_params.get("c0")
            c1 = kwargs.get("c1", None) or self.custom_params.get("c1")
            y_cutoff = kwargs.get("y_cutoff", None) or self.custom_params.get("y_cutoff")
            model_matrix = get_model_double_c(detx, dety, c0 / fs, c1 / fs,
                                              y_cutoff, dx, nx, x_0, nt, cache=cache)
        else:
            raise NotImplementedError(f"Model {model_type} is unavailable.")
        self._raw_model = model_matrix
        # Force dodgy duck typing to enable product with irf convolution.
        model_matrix.matvec = lambda x: None

        irf = kwargs.get("irf_model", None) or self.custom_params.get("irf_model")

        ndet = detx.shape[0]
        if irf is not None:
            convolve_irf = Convolve1D(nt * ndet, irf, dims=(ndet, nt), dir=1, offset=nt // 2,
                                      dtype=np.float32)

            def matvec(x):
                return convolve_irf @ (model_matrix @ x)

            def rmatvec(x):
                z = convolve_irf.H @ x
                return model_matrix.H @ z

            full_model = LinearOperator(model_matrix.shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float32)
            full_model.data = 1
        else:
            def matvec(x):
                return model_matrix @ x

            def rmatvec(x):
                return model_matrix.H @ x
            full_model = LinearOperator(model_matrix.shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float32)
            full_model.data = 1
        inv_args = {}

        if regulariser == "identity":
            reg = pylops.Identity(nx * nx)
        elif regulariser == "laplacian":
            if gpu:
                reg_mat = cp.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            else:
                reg_mat = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            reg = Convolve2D([nx, nx], reg_mat)
        elif regulariser == "TV":
            reg = [
                pylops.FirstDerivative(dims=(nx, nx), axis=0, edge=True, kind="backward", dtype=np.float32
                                       ),
                pylops.FirstDerivative(dims=(nx, nx), axis=1, edge=True, kind="backward", dtype=np.float32
                                       ),
            ]

            inv_args["RegsL2"] = [pylops.Identity(nx * nx)]
            inv_args["epsRL2s"] = [reg_lambda[1]]
            inv_args["mu"] = kwargs.get("mu", 1.)
            inv_args["tol"] = kwargs.get("tol", 1e-10)
            inv_args["tau"] = kwargs.get("tau", 1.)
        else:
            raise NotImplementedError(f"Regulariser {regulariser} is not available.")

        self._model_matrix = full_model, reg
        if regulariser in ["identity", "laplacian"]:
            if gpu:
                inv_args["niter"] = iter_lim
                inv_args["tol"] = 1e-6
            else:
                inv_args["iter_lim"] = iter_lim
            return lambda y: regularized_inversion(full_model, y,
                                                   [reg], epsRs=[reg_lambda], show=True, **inv_args)[0]
        else:
            self._full_model = full_model
            self._reg = reg
            self._niter_outer = iter_lim[0]
            self._niter_inner = iter_lim[1]
            self._epsRL1s = [reg_lambda[0], reg_lambda[0]]
            self._inv_args = inv_args
            return lambda y: pylops.optimization.sparsity.splitbregman(
                full_model,
                y,
                reg,
                niter_outer=iter_lim[0],
                niter_inner=iter_lim[1],
                epsRL1s=[reg_lambda[0], reg_lambda[0]],
                show=True, **inv_args
            )[0]

    def __init__(self,
                 n_pixels: Sequence[int],
                 field_of_view: Sequence[float],
                 kwargs_model=None,
                 **kwargs):
        super().__init__(n_pixels, field_of_view, **kwargs)
        self._model_matrix = None
        self._raw_model = None
        if kwargs_model is None:
            kwargs_model = {}
        # Note that super init puts kwargs in self.custom_params
        if "detectors" in kwargs:
            # identify x and y:
            detectors = kwargs["detectors"]
            indices = []
            for i in range(3):
                if np.all(kwargs["detectors"][:, i] != 0.):
                    indices.append(i)
            self._indices = indices

            detx = detectors[:, indices[0]]
            dety = detectors[:, indices[1]]

            fs = kwargs["fs_model"]
            self._fs = fs
            nx = n_pixels[indices[0]]
            self._nx = nx
            dx = field_of_view[indices[0]] / (nx - 1)
            self._dx = dx
            x_0 = -field_of_view[indices[0]] / 2
            self._x_0 = x_0
            nt = kwargs["nt"]

            gpu = kwargs.get("gpu", cuda_enabled)
            cache = kwargs.get("cache", False)

            self._model = self.generate_model(detx, dety, fs, dx, nx, x_0, nt, gpu=gpu, cache=cache, **kwargs_model)
        else:
            self._fs = None
            self._nx = None
            self._dx = None
            self._x_0 = None
            self._model = None
            self._indices = None

    def pre_prepare_data(self, x: PATimeSeries):
        """

        Parameters
        ----------
        x
        """
        # really just for speed of sound setting
        pass

    def reconstruct(self, signal: np.ndarray, fs: float,
                    geometry: np.ndarray,
                    n_pixels: Sequence[int],
                    field_of_view: Sequence[float],
                    speed_of_sound=None,
                    **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        signal
        fs
        geometry
        n_pixels
        field_of_view
        speed_of_sound
        kwargs

        Returns
        -------

        """
        gpu = self.custom_params.get("gpu", cuda_enabled)
        if self._model is not None:
            model = self._model
            fs = self._fs
            nx = self._nx
            dx = self._dx
            x_0 = self._x_0
            indices = self._indices
        else:
            detectors = geometry
            indices = []
            for i in range(3):
                if np.all(kwargs["detectors"][:, i] != 0.):
                    indices.append(i)
            detx = detectors[:, indices[0]]
            dety = detectors[:, indices[1]]

            do_irf = self.custom_params.get("do_irf", True)
            irf = kwargs["irf"] if do_irf else None

            nx = n_pixels[indices[0]]
            dx = field_of_view[indices[0]] / (nx - 1)
            x_0 = -field_of_view[indices[0]] / 2
            nt = signal.shape[-1]

            model = self.generate_model(detx, dety, fs, dx, nx, x_0, nt, irf_model=irf,
                                        gpu=gpu)
        if gpu:
            t = cp.array(signal)
        else:
            t = np.array(signal)

        or_shape = t.shape

        output_shape = [1, 1, 1]
        for i in indices:
            output_shape[i] = nx
        output_shape = tuple(output_shape)

        t = t.reshape((np.product(t.shape[:-2]),) + t.shape[-2:])
        output = np.zeros((t.shape[0],) + output_shape)

        for i, t0 in enumerate(t):
            result = model(t0.flatten()).reshape(output_shape)
            if gpu:
                output[i] = result.get()
            else:
                output[i] = result

        return output.reshape(or_shape[:2] + output.shape[-3:])

    @staticmethod
    def get_algorithm_name() -> str:
        """

        Returns
        -------

        """
        return "Model Based Reconstruction"
