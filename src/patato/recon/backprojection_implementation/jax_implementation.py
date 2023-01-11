#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def recon_partial(t, geometry, dl, nx, ny, nz, dx, dy, dz):
    """
    Do delay and sum for a single detector.
    """
    z, y, x = jnp.ogrid[0:nz, 0:ny, 0:nx]
    x = (x - (nx - 1) / 2) * dx
    y = (y - (ny - 1) / 2) * dy
    z = (z - (nz - 1) / 2) * dz
    offsets = (jnp.sqrt((geometry[0] - x) ** 2 +
                        (geometry[1] - y) ** 2 +
                        (geometry[2] - z) ** 2) / dl).astype(jnp.int32)
    return t[offsets]


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def full_recon(t, geometry, dl, nx, ny, nz, dx, dy, dz):
    """
    Do delay and sum for all detectors.
    """
    all_times = jax.vmap(recon_partial, in_axes=(0, 0,) + (None, ) * 7, out_axes=0)
    return jnp.sum(all_times(t, geometry, dl, nx, ny, nz, dx, dy, dz), axis=0)
