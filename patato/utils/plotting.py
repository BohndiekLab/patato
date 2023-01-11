#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ..io.attribute_tags import HDF5Tags


def animate_sequence(images, cmap, clim, **kwargs):
    fig, ax = plt.subplots()

    ims = []
    for i in range(images.shape[0]):
        im = ax.imshow(images[i], animated=True, cmap=cmap, clim=clim)
        if i == 0:
            ax.imshow(images[i], cmap=cmap, clim=clim)
        ims.append([im])

    default_kwargs = {"interval": 50, "blit": True, "repeat_delay": 1000}
    for d, k in default_kwargs.items():
        if d not in kwargs:
            kwargs[d] = k

    ani = animation.ArtistAnimation(fig, ims, **kwargs)
    return ani


type_cmaps = {HDF5Tags.SO2: "RdBu_r",
              HDF5Tags.THB: "magma",
              HDF5Tags.DELTA_SO2: "viridis",
              HDF5Tags.BASELINE_SO2: "RdBu_r",
              HDF5Tags.BASELINE_SO2_STANDARD_DEVIATION: "plasma",
              HDF5Tags.RECONSTRUCTION: "bone",
              HDF5Tags.DELTA_ICG: "cividis",
              "Responding Pixels": ListedColormap([(0, 0, 0, 0), "orange"]),
              HDF5Tags.RAW_DATA: "bone",
              None: "viridis",
              HDF5Tags.UNMIXED: "viridis"}
