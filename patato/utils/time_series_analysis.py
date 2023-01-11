#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import numpy as np
from patato.core.image_structures import image_structure_types
from scipy.signal import fftconvolve


def find_gc_boundaries(mask, data_so2,
                       window=10, display=False, sigma=2, skip_start=0,
                       sign=1):
    # Find the points at which the time series MSOT data switches from air to oxygen or oxygen to air.
    # STEP 1: Smooth the curve with a Gaussian kernel
    # Complete error here:
    measurement = image_structure_types.T
    if sign != 2:
        measurement *= sign

    kernel = np.arange(window) - window / 2 + 1 / 2
    kernel = np.exp(-(kernel / sigma) ** 2)
    kernel /= np.sum(kernel)
    kernel = kernel[:, None]

    from scipy.ndimage import median_filter
    smoothed = median_filter(measurement, kernel.shape)
    smoothed = fftconvolve(smoothed, kernel, "valid")

    # Find the gradient of the curve (to give us the maximum).
    smoothed_grad = np.median(np.gradient(smoothed, axis=0), axis=-1)
    steps = [skip_start]
    if sign == 2:
        smoothed_grad = smoothed_grad ** 2
    # Find first peak in derivative.
    step_point = np.argmax(smoothed_grad)
    steps.append(step_point + window // 2)
    step_grad = smoothed_grad[step_point]

    # Find second peak in the derivative
    step_point_b = step_point + np.argmin(smoothed_grad[step_point:])
    step_grad_b = smoothed_grad[step_point_b]

    # Custom stuff to try and avoid false positives (this could be improved somehow).
    if step_grad_b < - step_grad / 2 and step_point_b - step_point > step_point * 0.5:
        # I.e. detect if there actually was a second change.
        steps.append(step_point_b + window // 2)
        steps.append(len(measurement) - 1)
    else:
        steps.append(len(measurement) - 1)

    # Display the changeover points.
    if display:
        import matplotlib.pyplot as plt
        plt.plot(np.median(smoothed * sign, axis=-1), label="Gas Challenge Trace in the Reference Region")
        plt.twinx()
        plt.plot(smoothed_grad * sign, c="C1", label="Derivative of Gas Challenge Trace in the Reference Region")
        for s in steps:
            plt.axvline(s - window // 2)
        plt.gcf().legend()
        plt.show()
    return steps
