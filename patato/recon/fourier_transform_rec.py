#  Copyright (c) Thomas Else 2023.
#  License: MIT

from typing import Sequence

import numpy as np
from scipy.special import hankel1
from scipy.interpolate import RectBivariateSpline

from .reconstruction_algorithm import ReconstructionAlgorithm


def sin6hat(x):
    """ """
    res = (
        np.sin(6.0 * x) / 3.0
        - 3.0 * np.sin(4.0 * x)
        + 15.0 * np.sin(2.0 * x)
        - 20.0 * x
    )
    res = -res / (20.0 * np.pi)
    return res


def t6hat(rmax, rmin, x):
    """ """
    xx = np.abs(x)
    ones = np.ones_like(xx)
    xmax = rmax * ones
    xmin = rmin * ones

    xx1 = np.minimum(xx, xmax)
    xx2 = np.maximum(xx1, xmin)
    xx3 = (ones - (xx2 - rmin) / (rmax - rmin)) * np.pi
    xx4 = sin6hat(xx3)
    return xx4


class FFTReconstruction(ReconstructionAlgorithm):
    """
    Circular FFT-based reconstruction.

    Based on Python code by L. Kunyansky, University of Arizona.
    see: M. Eller, P. Hoskins, and L. Kunyansky
    Microlocally accurate solution of the inverse source problem
    of thermoacoustic tomography Inverse Problems 36(8), 2020, 094004
    """

    _batch = False

    def reconstruct(
        self,
        time_series: np.ndarray,
        fs: float,
        geometry: np.ndarray,
        n_pixels: Sequence[int],
        field_of_view: Sequence[float],
        speed_of_sound,
        **kwargs
    ) -> np.ndarray:
        shape = time_series.shape[:-2]
        time_series = time_series.reshape(
            (int(np.prod(shape)),) + time_series.shape[-2:]
        )
        output = []
        self.hankels = None
        for i in range(time_series.shape[0]):
            output.append(
                self._reconstruct(
                    time_series[i],
                    fs,
                    geometry,
                    n_pixels,
                    field_of_view,
                    speed_of_sound,
                    **kwargs
                )
            )
        o = np.stack(output)
        return o.reshape(shape + o.shape[-2:])

    def _reconstruct(
        self,
        raw_timeseries_data: np.ndarray,
        fs: float,
        geometry: np.ndarray,
        n_pixels: Sequence[int],
        field_of_view: Sequence[float],
        speed_of_sound,
        **kwargs
    ) -> np.ndarray:
        """
        Parameters
        ----------
        time_series
        fs
        geometry
        n_pixels
        field_of_view
        speed_of_sound
        kwargs

        Returns
        -------


        """
        n_grid_detectors = kwargs.get("n_grid_detectors", 1024)
        n_samples_padded_grid = kwargs.get(
            "n_samples_padded_grid", max(4096, raw_timeseries_data.shape[-1])
        )

        return_ft = kwargs.get("return_ft", False)
        debug = kwargs.get("debug", False)

        # Validate input
        detector_radii = np.linalg.norm(geometry, axis=1)
        detector_radius = detector_radii[0]
        nonzero_axes = [
            i for i in range(geometry.shape[-1]) if not np.all(geometry[:, i] == 0)
        ]

        if not np.all(np.isclose(detector_radii, detector_radius)):
            raise ValueError(
                "All points on detector must be on a circle centred around (0, 0)."
            )
        if not len(nonzero_axes) == 2:
            raise ValueError(
                "Detectors must be 2-dimensional, i.e. geometry array must either be (ndet, 2) or (ndet, 3) with np.all(geometry[:, i] == 0) for a single value of i."
            )
        geometry = geometry[:, nonzero_axes]

        field_of_view = [
            f for i, f in enumerate(field_of_view) if n_pixels[i] not in [0, 1]
        ]
        n_pixels = [f for f in n_pixels if f not in [0, 1]]

        if (
            not len(field_of_view) == 2
            or not len(n_pixels) == 2
            or field_of_view[1] != field_of_view[0]
            or n_pixels[0] != n_pixels[1]
        ):
            raise ValueError()

        image_pixels = int(n_pixels[0])
        image_width = field_of_view[0]
        c = speed_of_sound

        mean_angle = np.arctan2(np.mean(geometry[:, 1]), np.mean(geometry[:, 0]))
        detector_angles = (np.arctan2(geometry[:, 1], geometry[:, 0]) - mean_angle) % (
            2 * np.pi
        )
        detector_angles[detector_angles > np.pi] = (
            detector_angles[detector_angles > np.pi] - np.pi * 2
        )

        nt = raw_timeseries_data.shape[-1]
        ndet = raw_timeseries_data.shape[-2]

        # 1. Crop the time series to exclude early and late bits.
        early_cut = (np.sqrt(2) * image_width / 2) / (c / fs) - 50
        late_cut = (detector_radius + np.sqrt(2) * image_width / 2) / (c / fs) + 50

        # TODO: You can add zero padding here too in future?
        hatfunction = t6hat(min(late_cut + 400, nt), late_cut, np.arange(nt))
        hatfunction *= 1 - t6hat(early_cut, early_cut - 400, np.arange(nt))

        if debug:
            import matplotlib.pyplot as plt

            plt.title("What does t6hat do?")
            plt.plot(raw_timeseries_data[128], label="Timeseries")
            plt.plot(
                raw_timeseries_data[128] * hatfunction, label="Smooth timeseries cut"
            )
            plt.plot(
                hatfunction * np.max(raw_timeseries_data[128]),
                c="C2",
                label="Hat function",
            )
            plt.legend(frameon=False)
            plt.show()
        timeseries_data = raw_timeseries_data * hatfunction

        # 2. Interpolate the data onto an angular grid.
        new_angles_detectors = np.linspace(
            -np.pi, np.pi, n_grid_detectors, endpoint=False
        )

        timeseries = np.zeros((n_grid_detectors, n_samples_padded_grid))
        # NP.INTERP requires sorted arguments.

        assert np.all(new_angles_detectors % (2 * np.pi) >= 0)
        timeseries[:, : timeseries_data.shape[1]] = np.stack(
            [
                np.interp(
                    new_angles_detectors, detector_angles, timeseries_data[:, i], 0, 0
                )
                for i in range(nt)
            ]
        ).T

        if debug:
            plt.imshow(
                timeseries,
                extent=(0, timeseries.shape[1], -np.pi, np.pi),
                aspect="auto",
            )
            plt.xlabel("Time samples")
            plt.ylabel("Detector angle (rad)")
            plt.title("Pre-processed time series")
            plt.show()

        # 3. Fourier Transform
        nfft_positive = (
            n_samples_padded_grid // 2
            if n_samples_padded_grid % 2 == 1
            else n_samples_padded_grid // 2 - 1
        )
        ft_timeseries = np.fft.fftshift(np.fft.ifft2(timeseries), 0)[:, :nfft_positive]

        if debug:
            plt.imshow(np.log(np.abs(ft_timeseries)), aspect="auto")
            plt.xlabel("Time frequency (index)")
            plt.ylabel("Angle frequency (index)")
            plt.title("FT of time series")
            plt.show()

        time_frequencies = 2 * np.pi * np.fft.fftfreq(n_samples_padded_grid, d=1 / fs)
        positive_time_frequencies = time_frequencies[:nfft_positive]
        freq_rad = time_frequencies[:nfft_positive] * detector_radius / c

        # 4. Compute Hankel functions for division later
        if self.hankels is None:
            indices_detector_grid = np.arange(n_grid_detectors // 2)
            indices, frequencies = np.meshgrid(
                indices_detector_grid, freq_rad[1:], indexing="ij"
            )

            hankel_temp = hankel1(indices, frequencies) * frequencies
            hankel_temp[np.isnan(hankel_temp)] = 1e30
            hankel_temp[np.abs(hankel_temp) > 1e30] = 1e30

            hankels = np.zeros((n_grid_detectors, nfft_positive), dtype=np.complex128)
            hankels[-n_grid_detectors // 2 :, 1:] = hankel_temp
            hankels[1 : n_grid_detectors // 2, 1:] = hankel_temp[::-1][
                : n_grid_detectors // 2 - 1
            ]

            hankels[:, 0] = 1e30
            hankels[0] = 1e30

            if hankels.shape[0] % 2 == 1:
                hankels[hankels.shape[0] // 2] = 1e30
        else:
            hankels = self.hankels

        # 5. Apply Hankel multiplication etc.
        k = np.abs(np.arange(n_grid_detectors) - n_grid_detectors // 2)
        coef = (-1j) ** k
        ft_timeseries = ft_timeseries / coef[:, None]
        ft_timeseries = ft_timeseries / hankels

        zero_frequency_index = n_grid_detectors // 2
        j1 = hankels[zero_frequency_index + 1].real
        csum = np.sum(
            ft_timeseries[zero_frequency_index, 1:] * j1[1:] / freq_rad[1:]
        ) * (freq_rad[1] - freq_rad[0])

        # 6. Fourier transform the detector axis to angles
        tran = np.fft.fft(np.fft.fftshift(ft_timeseries, 0), axis=0)
        tran[:, 0] = csum.real

        # 7. Interpolate onto a Cartesian grid.
        dx_time = (image_width / (image_pixels - 1)) / c
        k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(image_pixels * 3, dx_time))

        kx, ky = np.meshgrid(k, k)
        freq_rho = np.sqrt(kx**2 + ky**2)
        freq_theta = np.arctan2(ky, kx) % (2 * np.pi)

        if debug:
            plt.imshow(np.log(np.abs(tran)))
            plt.show()
        # We're duplicating the data at theta = 0 to 2pi to allow interpolation on a circle.
        tran_ext = np.zeros((tran.shape[0] + 1, tran.shape[1]), dtype=tran.dtype)
        tran_ext[: tran.shape[0]] = tran
        tran_ext[-1] = tran[0]
        if debug:
            plt.imshow(np.log(np.abs(tran_ext)))
            plt.show()
        interp_spline_r = RectBivariateSpline(
            np.linspace(0, 2 * np.pi, n_grid_detectors + 1),
            positive_time_frequencies,
            tran_ext.real,
        )
        interp_spline_i = RectBivariateSpline(
            np.linspace(0, 2 * np.pi, n_grid_detectors + 1),
            positive_time_frequencies,
            tran_ext.imag,
        )

        cart_transform = interp_spline_r(
            freq_theta, freq_rho, grid=False
        ) + 1j * interp_spline_i(freq_theta, freq_rho, grid=False)

        mcenter = kx.shape[0] // 2
        lhalf = kwargs.get("lhalf", 3)  # TODO: Work out when to use different values?
        if lhalf == 1:
            cart_transform[:mcenter, :] = np.conj(
                np.copy(np.flip(cart_transform[-mcenter:, :], [0, 1]))
            )
        if lhalf == 2:
            cart_transform[-mcenter:, :] = np.conj(
                np.copy(np.flip(cart_transform[:mcenter, :], [0, 1]))
            )
        if lhalf == 3:
            cart_transform[:, :mcenter] = np.conj(
                np.copy(np.flip(cart_transform[:, -mcenter:], [0, 1]))
            )
        if lhalf == 4:
            cart_transform[:, -mcenter:] = np.conj(
                np.copy(np.flip(cart_transform[:, :mcenter], [0, 1]))
            )

        image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(cart_transform)).real)
        image = image[image_pixels : image_pixels * 2, image_pixels : image_pixels * 2]

        # align with standard
        image = np.flipud(image.T)
        if return_ft:
            return image, cart_transform, k
        return image

    @staticmethod
    def get_algorithm_name() -> str:
        """
        Returns
        -------


        """
        return "Reference Backprojection"
