#  Copyright (c) Janek Grohl 2023.
#  License: MIT

from typing import Tuple, Optional, List, Union

import numpy as np

from learned_PA_sO2_estimator.utils.io import convert_numpy_array
from learned_PA_sO2_estimator.estimate_sO2 import evaluate

from ..core.image_structures.reconstruction_image import Reconstruction
from ..core.image_structures.single_parameter_data import SingleParameterData
from ..processing.processing_algorithm import SpatialProcessingAlgorithm


class LearnedSpectralUnmixer(SpatialProcessingAlgorithm):
    """
    The LearnedSpectralUnmixer class can be used to unmix multispectral photoacoustic data with a machine learning-based
    unmixing algorithm that is trained on a specific training data set.

    This takes in reconstruction data and spits out spectrally unmixed data.
    """

    @staticmethod
    def re_grid(reconstruction: np.ndarray, scaling_factor: int) -> np.ndarray:
        """
        This is copy-pasted from the linear unmixer class.
        There might be a good case for writing an abstract base class to avoid redundant code.

        This rescaling applies a smoothing function by convolving the image with a kernel of ones.
        Then it extracts N equidistant slices from the original image, where N is calculated from the scaling factor.

        The blurring is introduced to prevent violating nyquist theorem by removing high frequencies from the image.

        Parameters
        ----------
        reconstruction
        scaling_factor

        Returns
        -------
        a numpy array that is rescaled.

        """
        if scaling_factor == 1:
            return reconstruction
        else:
            n_spatial = 3 if reconstruction.ndim == 5 else 2
            reshape_kernel_size = tuple(min(scaling_factor, size) for size in reconstruction.shape[-n_spatial:])
            extend = (None,) * len(reconstruction.shape[:-n_spatial])
            kernel = np.ones(reshape_kernel_size)[extend]
            kernel /= np.sum(kernel)
            from scipy.signal import convolve
            smoothed = convolve(reconstruction, kernel, mode="same")
            slice_selection = (slice(None, None),) * len(reconstruction.shape[:-n_spatial])
            slice_selection += (slice(None, None, scaling_factor),) * n_spatial
            return smoothed[slice_selection]

    def run(self, reconstruction: Reconstruction, _, **kwargs) -> Tuple[SingleParameterData, dict, None]:
        # Select the right wavelengths:
        wavelengths = self.wavelengths
        wavelength_indices = np.where(np.isclose(reconstruction.wavelengths[:, None], wavelengths[None, :]))[0]

        # Get the reconstructed data
        recon_data = reconstruction[:, wavelength_indices].raw_data

        # Change the grid
        recon_data = self.re_grid(recon_data, self.rescaling_factor)

        original_shape = np.shape(recon_data)
        n_frames = original_shape[0]
        n_wl = original_shape[1]

        # reshape recon_data to (n_samples, n_wavelengths) from [frames, wavelengths, x, y, z]
        recon_data = np.swapaxes(recon_data, 0, 1)
        recon_data = np.reshape(recon_data, (n_wl, -1))
        recon_data = np.swapaxes(recon_data, 0, 1)

        # Unmix.
        recon_data_input = convert_numpy_array(recon_data, wavelengths)
        unmixed = evaluate(recon_data_input, self.train_dataset_id, len(wavelengths))

        # from (n_samples, n_wavelengths) to [frames, wavelengths, x, y, z]
        unmixed = np.reshape(unmixed, (n_frames, -1, 1))
        unmixed = np.swapaxes(unmixed, 1, 2)
        new_shape = np.asarray(original_shape)
        new_shape[1] = 1
        unmixed = np.reshape(unmixed, new_shape)

        output_data = SingleParameterData(unmixed.astype(np.float32), ["learned_sO2"],
                                          algorithm_id=self.algorithm_id,
                                          attributes=reconstruction.attributes,
                                          field_of_view=reconstruction.fov_3d)
        output_data.attributes["TrainingDataset"] = self.train_dataset_id
        output_data.attributes["wavelengths"] = self.wavelengths
        output_data.hdf5_sub_name = reconstruction.hdf5_sub_name
        return output_data, {}, None

    def __init__(self, train_dataset_id: int, wavelengths, rescaling_factor=1):
        super().__init__("learned_sO2")
        self.train_dataset_id = train_dataset_id
        self.wavelengths = np.array(wavelengths)
        self.rescaling_factor = rescaling_factor
