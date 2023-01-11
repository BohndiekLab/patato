#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import numpy as np
from scipy.sparse import csr_matrix, vstack

from os.path import join, exists, dirname
from .generate_model import calculate_element
from tqdm import tqdm


def get_hash(*x):
    to_hash = []
    for y in x:
        if type(y) == np.ndarray:
            y = tuple(y.flatten())
        to_hash.append(y)
    h = hash(tuple(to_hash))
    return hex(np.uint64(h))


def generate_model(det_x, det_y, dl_0, dx, nx, x_0, nt):
    """

    Parameters
    ----------
    det_x
    det_y
    dl_0
    dl_1
    y_cutoff
    dx
    nx
    x_0
    nt

    Returns
    -------

    """
    # TODO: validate types.
    # TODO: allow non-square reconstruction areas.
    ntpixel = np.int32(4 * np.sqrt(2) * dx / dl_0)

    # Normalise the values. Convert to appropriate format.
    det_x = np.double(det_x)
    det_y = np.double(det_y)
    x_0 = np.double(x_0)
    dx = np.double(dx)

    matrices = []
    i = 0

    output = np.zeros((ntpixel * nx * nx), dtype=np.float64)
    indices = np.zeros((ntpixel * nx * nx), dtype=np.int32)

    positions = np.repeat(np.arange(nx * nx)[:, None], ntpixel, axis=-1).flatten()

    for detx, dety in tqdm(list(zip(det_x, det_y))):
        i += 1

        calculate_element(output, indices, nx, ntpixel, detx, dety, dl_0, x_0, dx)

        matrix = csr_matrix((output.flatten(), (indices.flatten(), positions)), shape=(nt, nx * nx))
        matrices.append(matrix)
    m = vstack(matrices)
    return m


def get_model(det_x, det_y, dl, dx, nx, x_0, nt,
              cache=True, hash_fn=None):
    det_x = det_x.astype(np.float64)
    det_y = det_y.astype(np.float64)
    dl = np.float64(dl)
    dx = np.float64(dx)
    nx = np.int32(nx)
    x_0 = np.float64(x_0)
    nt = np.int32(nt)
    print("Loading model")
    if hash_fn is None:
        hash_fn = get_hash
    if cache:
        h = hash_fn(det_x, det_y, dl, dx, nx, x_0, nt)
        model_folder = join(dirname(__file__), "models")
        filename = join(model_folder, h + ".npz")
        import scipy.sparse
        if exists(filename):
            mat = csr_matrix(scipy.sparse.load_npz(filename))
        else:
            mat = generate_model(det_x, det_y, dl, dx, nx, x_0, nt)
            scipy.sparse.save_npz(filename, mat.astype(np.float32).get(), compressed=False)
        return mat
    else:
        return generate_model(det_x, det_y, dl, dx, nx, x_0, nt)
