#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import logging
import os
from typing import Sequence

import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array
except ImportError:
    cl = None
    pyopencl = None

from .reconstruction_algorithm import ReconstructionAlgorithm
from ..core.image_structures.pa_time_data import PATimeSeries


def validate_opencl_input(signal: np.ndarray, fs, detectors, n_pixels, field_of_view, speed_of_sound):
    # Make sure that the input to the OpenCL code is valid.
    if type(signal) is not np.ndarray:
        signal = np.asarray(signal)
    if type(detectors) is not np.ndarray:
        detectors = np.asarray(detectors)

    if not signal.flags["C_CONTIGUOUS"]:
        signal = signal.copy()
        logging.warning("Warning: order of signal is not C_CONTIGUOUS, this might hinder performance.")
    if not (type(fs) == float or type(fs) in [np.float64, np.float32]):
        fs = float(fs)
    if not detectors.flags["C_CONTIGUOUS"]:
        detectors = detectors.copy()
    assert type(n_pixels[0]) in [int, np.int32]
    assert type(field_of_view[0]) in [float, np.float32, np.float64]
    if not (type(speed_of_sound) in [float, np.float32, np.float64]):
        speed_of_sound = float(speed_of_sound)
    return signal, fs, detectors, n_pixels, field_of_view, speed_of_sound


class OpenCLBackprojection(ReconstructionAlgorithm):
    """
    Note: this class will be deprecated in future. It is maintained for backwards compatibility with non-nvidia
    GPUs.
    """
    def __init__(self,
                 n_pixels: Sequence[int],
                 field_of_view: Sequence[float],
                 **kwargs):
        self.ctx = kwargs.get("ctx", None)
        self.queue = kwargs.get("queue", None)
        if self.ctx is None:
            self.ctx = pyopencl.create_some_context(interactive=False)
            self.queue = pyopencl.CommandQueue(self.ctx)

        # Connect to the OpenCL code:
        directory = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(directory, "opencl_code/dascl.cl")) as file:
            code = file.read()

        program = cl.Program(self.ctx, code).build()
        cldas = program.cldas
        cldas.set_scalar_arg_dtypes([None, None, None, np.float32, np.intc, np.intc, np.intc,
                                     np.float32, np.float32, np.float32, np.float32, np.float32,
                                     np.float32, np.intc, np.intc, None, np.intc, np.intc])
        super().__init__(n_pixels, field_of_view, ctx=self.ctx, queue=self.queue, cldas=cldas)

    def pre_prepare_data(self, x: PATimeSeries):
        # really just for speed of sound setting
        return x.to_opencl(self.queue)

    def reconstruct(self, signal, fs, detectors, n_pixels: Sequence[int],
                    field_of_view: Sequence[float],
                    speed_of_sound=None,
                    queue=None,
                    cldas=None,
                    **kwargs) -> np.ndarray:
        signal, fs, detectors, n_pixels, field_of_view, speed_of_sound = validate_opencl_input(signal, fs, detectors,
                                                                                               n_pixels, field_of_view,
                                                                                               speed_of_sound)
        if type(signal) is pyopencl.array.Array and queue is None:
            queue = signal.queue
        if queue is None:
            ctx = cl.create_some_context(interactive=False)
            queue = cl.CommandQueue(ctx)
        if cldas is None:
            raise ValueError()

        if type(signal) is not pyopencl.array.Array:
            signal = pyopencl.array.to_device(queue, signal.astype(np.float32))

        logging.info(f"{detectors.flags}")
        if type(detectors) is not pyopencl.array.Array:
            detectors = pyopencl.array.to_device(queue, detectors.copy().astype(np.float32))

        logging.info(f"{signal.shape}, {speed_of_sound}, {fs}, {detectors.shape}")

        original_shape = signal.shape[:-2]
        frames = int(np.product(original_shape))
        signal = signal.reshape((frames,) + signal.shape[-2:])
        dl = speed_of_sound / fs

        logging.info(f"{fs}, {speed_of_sound}.")

        nx, ny, nz = n_pixels

        if nx == 1:
            dx = 1
        else:
            dx = field_of_view[0] / (n_pixels[0] - 1)

        if ny == 1:
            dy = 1
        else:
            dy = field_of_view[1] / (n_pixels[1] - 1)

        if nz == 1:
            dz = 1
        else:
            dz = field_of_view[2] / (n_pixels[2] - 1)

        x_0 = - field_of_view[0] / 2 if nx != 1 else 0
        y_0 = - field_of_view[1] / 2 if ny != 1 else 0
        if nz == 1:
            z_0 = 0
        else:
            z_0 = - field_of_view[2] / 2

        n_detectors = detectors.shape[0]
        n_samples = signal.shape[-1]

        max_size = min(queue.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE), n_detectors)
        local_memory = cl.LocalMemory(max_size * 4)
        frame_offset = int(signal.offset // (n_detectors * n_samples * 4))
        output = pyopencl.array.zeros(queue, (frames, n_detectors // max_size) + tuple(n_pixels), dtype=signal.dtype)

        logging.info(f"OpenCL debugging: {queue}, {nx * ny * nz}, {n_detectors}, {max_size}, {output.dtype}, "
                     f"{signal.dtype}, {detectors.dtype}, {dl}, {nx}, {ny}, {nz}, {x_0}, "
                     f"{y_0}, {z_0}, {dx}, {dy}, {dz}, {n_detectors}, {n_samples}, "
                     f"{local_memory}, {frames}, {frame_offset}, {signal.shape}, {original_shape}.")
        for f in range(frames):
            logging.info(f"Step {f}.")
            try:
                cldas(queue, (nx * ny * nz, n_detectors), (1, max_size),
                      output.data, signal.base_data, detectors.data, dl,
                      nx, ny, nz, x_0, y_0, z_0, dx, dy, dz, n_detectors,
                      n_samples, local_memory,
                      f, frame_offset)
            except pyopencl.LogicError:
                max_size = 1
                local_memory = cl.LocalMemory(max_size * 4)
                output = pyopencl.array.zeros(queue, (frames, n_detectors // max_size) + tuple(n_pixels),
                                              dtype=signal.dtype)
                cldas(queue, (nx * ny * nz, n_detectors), (1, max_size),
                      output.data, signal.base_data, detectors.data, dl,
                      nx, ny, nz, x_0, y_0, z_0, dx, dy, dz, n_detectors,
                      n_samples, local_memory,
                      f, frame_offset)
            queue.finish()

        numpy_output = np.sum(output.get(), axis=1)
        try:
            output.base_data.release()
        except AttributeError:
            logging.debug("OpenCL can't release base data, continuing anyway...")
        queue.flush()
        return numpy_output.reshape(original_shape + tuple(n_pixels)[::-1])

    @staticmethod
    def get_algorithm_name() -> str:
        return "OpenCL Backprojection"
