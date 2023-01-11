try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix, vstack
    cuda_enabled = True
except ImportError:
    cuda_enabled = False
#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from os.path import dirname, join, exists

import numpy as np
from .cuda_implementation import get_hash


def generate_model(det_x, det_y, dl_0, dl_1, y_cutoff, dx, nx, x_0, nt):
    # TODO: validate types.
    # Load the cuda code:
    directory = dirname(__file__)

    with open(join(directory, "generate_model_refraction.cu"), "r") as w:
        cuda_code = w.read()
    calculate_element = cp.RawKernel(cuda_code, 'calculate_element', jitify=True)

    # TODO: allow non-square reconstruction areas.
    ntpixel = cp.int32(4 * np.sqrt(2) * dx / min(dl_0, dl_1))

    # Normalise the values. Convert to appropriate format.
    det_x = cp.double(det_x)
    det_y = cp.double(det_y)
    x_0 = cp.double(x_0)
    dx = cp.double(dx)

    matrices = []
    i = 0

    output = cp.zeros(int(ntpixel * nx * nx), dtype=cp.float64)
    indices = cp.zeros(int(ntpixel * nx * nx), dtype=cp.int32)

    positions = cp.repeat(cp.arange(nx * nx)[:, None], ntpixel, axis=-1).flatten()

    for detx, dety in zip(det_x, det_y):
        i += 1
        # TODO: optimise the block/grid size below.
        calculate_element((128,), (128,), (output, indices, nx,
                                           ntpixel,
                                           detx, dety, dl_0, dl_1, x_0, dx, y_cutoff))

        matrix = csr_matrix((output.flatten(), (indices.flatten(), positions)), shape=(nt, nx * nx))
        matrices.append(matrix)
    m = vstack(matrices)
    return m


def get_model(det_x, det_y, dl_0, dl_1, y_cutoff, dx, nx, x_0, nt,
              cache=True, hash_fn=None):
    det_x = det_x.astype(np.float64)
    det_y = det_y.astype(np.float64)
    dl_0 = cp.float64(dl_0)
    dl_1 = cp.float64(dl_1)
    dx = cp.float64(dx)
    nx = cp.int32(nx)
    x_0 = cp.float64(x_0)
    y_cutoff = cp.float64(y_cutoff)
    nt = cp.int32(nt)
    print("Loading model")
    if hash_fn is None:
        hash_fn = get_hash
    if cache:
        h = hash_fn(det_x, det_y, dl_0, dl_1, y_cutoff, dx, nx, x_0, nt)
        model_folder = join(dirname(__file__), "models")
        filename = join(model_folder, h + ".npz")
        import scipy.sparse
        if exists(filename):
            mat = csr_matrix(scipy.sparse.load_npz(filename))
        else:
            mat = generate_model(det_x, det_y, dl_0, dl_1, y_cutoff, dx, nx, x_0, nt)
            scipy.sparse.save_npz(filename, mat.astype(cp.float32).get(), compressed=False)
        return mat
    else:
        return generate_model(det_x, det_y, dl_0, dl_1, y_cutoff, dx, nx, x_0, nt)
