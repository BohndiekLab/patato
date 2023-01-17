#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import numpy as np
from scipy.signal import fftconvolve


def find_gc_boundaries(mask, data_so2,
                       window=10, display=False, sigma=2, skip_start=0,
                       sign=1):
    """
    Find the runs in the time series at which the breathing gas of the animal was changed.

    Parameters
    ----------
    mask : array_like
        A boolean array which can be applied to the time series.
    data_so2 : SingleParameterData
        The time series of the SO2 data.
    window : int, optional
        The window size to use for the convolution.
    display : bool, optional
        Whether to display the results.
    sigma : float, optional
        The sigma to use for the convolution.
    skip_start : int, optional
        The number of runs to skip at the start of the time series.
    sign : int, optional
        The sign of the change in the time series.

    Returns
    -------
    list
        List of integers with the indices of the detected gas changes.
    """

    # STEP 1: Smooth the curve with a Gaussian kernel
    # Complete error here:
    measurement = np.squeeze(data_so2.raw_data.T[mask.T].T)

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
