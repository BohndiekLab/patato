#  Copyright (c) Thomas Else 2023.
#  License: BSD-3


import numpy as np


def get_basic_p0(nx=333, dx=75e-6, r=0.005):
    """

    Get a basic p0 for the simulation.
    Parameters
    ----------
    nx : int
    dx : float

    Returns
    -------
    p0 : ndarray

    """

    x, y = np.ogrid[:nx, :nx]
    x = x.astype(np.float64) - (nx - 1) / 2
    y = y.astype(np.float64) - (nx - 1) / 2
    x = x * dx
    y = y * dx
    r2 = r ** 2
    p0 = r2 - (x ** 2 + y ** 2)
    p0[p0 < 0] = 0
    return p0


def generate_basic_model(ndet=256):
    """
    Generate a basic forward model for a photoacoustic system.

    Parameters
    ----------
    ndet

    Returns
    -------
    tuple of array_like, array_like
    """
    try:
        import cupy as cp
        from ..recon.model_based.cuda_implementation import get_model
        CUDA_ENABLED = True
    except ImportError:
        from ..recon.model_based.numpy_implementation import get_model
        CUDA_ENABLED = False

    theta = np.linspace(0, 2 * np.pi, ndet, endpoint=False)
    detectors = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).T * 0.04

    model = get_model(detectors[:, 0], detectors[:, 1], 1500 / 4e7, 75e-6, 333, -0.0125, 2030, cache=False)
    if CUDA_ENABLED:
        model = model.get()
    model.eliminate_zeros()
    model.sort_indices()
    return model, detectors


def generate_basic_simulation(ndet=256):
    """
    Generate a simulation of the photoacoustic time series for a given number of detectors in a circular geometry.
    The initial pressure distribution used is a truncated paraboloid, which has an analytical solution for the
    time series data. Here, we use the forward photacoustic model from the model-based reconstruction code.

    Parameters
    ----------
    ndet: int - Number of detectors

    Returns
    -------
    tuple of NDArray, NDArray, NDArray

    """
    model, detectors = generate_basic_model(ndet)
    p0 = get_basic_p0()
    ts = (model @ p0.flatten()).reshape(ndet, 2030)
    return p0, ts, detectors
