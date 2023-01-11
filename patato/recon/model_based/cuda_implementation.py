try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix, vstack
    cupy_enabled = True
except ImportError:
    cupy_enabled = False

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from os.path import dirname, join, exists

import numpy as np


def generate_model(det_x, det_y, dl, dx, nx, x_0, nt):
    # TODO: validate types.
    # Load the cuda code:
    directory = dirname(__file__)

    with open(join(directory, "generate_model.cu"), "r") as w:
        cuda_code = w.read()
    calculate_element = cp.RawKernel(cuda_code, 'calculate_element', jitify=True)

    # TODO: allow non-square reconstruction areas.
    ntpixel = cp.int32(4 * np.sqrt(2) * dx / dl)

    # Normalise the values. Convert to appropriate format.
    det_x /= dl
    det_x = cp.double(det_x)
    det_y /= dl
    det_y = cp.double(det_y)
    x_0 /= dl
    x_0 = cp.double(x_0)
    dx /= dl
    dx = cp.double(dx)

    matrices = []
    i = 0

    output = cp.zeros(int(ntpixel * nx * nx), dtype=cp.float64)
    indices = cp.zeros(int(ntpixel * nx * nx), dtype=cp.int32)

    positions = cp.repeat(cp.arange(nx * nx)[:, None], ntpixel, axis=-1).flatten()

    dl = cp.float64(1.)
    for detx, dety in zip(det_x, det_y):
        i += 1
        # TODO: optimise the block/grid size below.
        calculate_element((128,), (128,), (output, indices, nx,
                                           ntpixel,
                                           detx, dety, dl, x_0, dx))
        matrix = csr_matrix((output.flatten(), (indices.flatten(), positions)), shape=(nt, nx * nx))
        matrices.append(matrix)
    m = vstack(matrices)
    return m


def get_hash(*x):
    to_hash = []
    for y in x:
        if type(y) == np.ndarray:
            y = tuple(y.flatten())
        to_hash.append(y)
    h = hash(tuple(to_hash))
    return hex(np.uint64(h))


def get_model(det_x, det_y, dl, dx, nx, x_0, nt,
              cache=True, hash_fn=None):
    det_x = det_x.astype(np.float64)
    det_y = det_y.astype(np.float64)
    dl = cp.float64(dl)
    dx = cp.float64(dx)
    nx = cp.int32(nx)
    x_0 = cp.float64(x_0)
    nt = cp.int32(nt)
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
            scipy.sparse.save_npz(filename, mat.astype(cp.float32).get(), compressed=False)
        return mat
    else:
        return generate_model(det_x, det_y, dl, dx, nx, x_0, nt)


def test_forward_model(hash_fn=None):
    import matplotlib.pyplot as plt
    dl = cp.float64(1540 / 4e7)

    det_thetas = np.linspace(cp.pi / 4, 7 * cp.pi / 4, 256)
    detectors = np.array([np.cos(det_thetas), np.sin(det_thetas)]).T * 0.0405

    lx = cp.float64(0.025)
    x_0 = -lx / 2
    nx = 512
    dx = lx / (nx - 1)
    nt = 2030

    print("Generating model.")
    model = get_model(detectors[:, 0], detectors[:, 1], dl, dx, nx, x_0, nt, hash_fn=hash_fn)
    print("Model generation complete.")

    x = cp.linspace(-1, 1, nx, dtype=cp.float32)
    xs, ys = cp.meshgrid(x, x)

    p = 0.5 - (xs ** 2 + ys ** 2)
    p[p < 0] = 0

    plt.imshow(p.get())
    plt.show()

    timeseries = (model @ p.flatten()).get().reshape(-1, nt)

    plt.imshow(timeseries, aspect="auto")
    plt.show()

    y = timeseries[0:1, :]
    plt.plot(y.T)
    m1 = np.min(np.where(y != 0)[-1]) - 10
    m2 = np.max(np.where(y != 0)[-1]) + 10
    plt.xlim(m1, m2)
    plt.show()
