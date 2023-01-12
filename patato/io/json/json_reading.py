#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import json
from typing import Union, Optional

from ..attribute_tags import PreprocessingAttributeTags, UnmixingAttributeTags, ReconAttributeTags
from ..msot_data import PAData
from ...processing.preprocessing_types import PREPROCESSING_NAMES
from ...unmixing.unmixer import SpectralUnmixer, SO2Calculator, THbCalculator
from ...recon import RECONSTRUCTION_NAMES


def read_reconstruction_preset(json_path: Union[str, dict]):
    """
        Load and parse the reconstruction preset from the specified JSON file or dictionary.

        Parameters
        ----------
        json_path : Union[str, dict]
            The path to the JSON file or a dictionary containing the reconstruction preset.

        Returns
        -------
        settings : SpatialProcessingAlgorithm
            The parsed reconstruction preset as a pipeline element.

        """
    # Load json
    if type(json_path) == str:
        with open(json_path) as json_file:
            settings = json.load(json_file)
    else:
        settings = json_path

    # Preprocessing parameters
    filter_highpass = settings.get(PreprocessingAttributeTags.HIGH_PASS_FILTER, None)
    filter_lowpass = settings.get(PreprocessingAttributeTags.LOW_PASS_FILTER, None)

    time_interpolation_factor = settings.get(PreprocessingAttributeTags.TIME_INTERPOLATION, 3)
    detector_interpolation_factor = settings.get(PreprocessingAttributeTags.DETECTOR_INTERPOLATION, 2)

    correct_for_impulse_response = settings.get(PreprocessingAttributeTags.IMPULSE_RESPONSE, True)
    do_hilbert_transform = settings.get(PreprocessingAttributeTags.HILBERT_TRANSFORM, False)
    do_envelope_detection = settings.get(PreprocessingAttributeTags.ENVELOPE_DETECTION, False)

    window_size = settings.get(PreprocessingAttributeTags.WINDOW_SIZE, 512)

    absolute_transformation = "imag"
    if do_envelope_detection:
        absolute_transformation = "abs"
    elif not do_hilbert_transform:
        absolute_transformation = "real"

    preprocessing_algorithm = PREPROCESSING_NAMES[settings[PreprocessingAttributeTags.PROCESSING_ALGORITHM]]

    # Reconstruction parameters.
    recon_params = settings.get(ReconAttributeTags.ADDITIONAL_PARAMETERS, {})  # Custom parameters

    # Field of view y and z:
    field_of_view_x = settings.get(ReconAttributeTags.X_FIELD_OF_VIEW, 0.)
    field_of_view_y = settings.get(ReconAttributeTags.Y_FIELD_OF_VIEW, 0.)
    field_of_view_z = settings.get(ReconAttributeTags.Z_FIELD_OF_VIEW, 0.)

    number_of_pixels_x = settings.get(ReconAttributeTags.X_NUMBER_OF_PIXELS, 1)
    number_of_pixels_y = settings.get(ReconAttributeTags.Y_NUMBER_OF_PIXELS, 1)
    number_of_pixels_z = settings.get(ReconAttributeTags.Z_NUMBER_OF_PIXELS, 1)

    # Number of pixels x and y
    reconstruction_algorithm = RECONSTRUCTION_NAMES[settings[ReconAttributeTags.RECONSTRUCTION_ALGORITHM]]

    step_1 = preprocessing_algorithm(time_factor=time_interpolation_factor,
                                     detector_factor=detector_interpolation_factor,
                                     irf=correct_for_impulse_response,
                                     hilbert=do_hilbert_transform,
                                     lp_filter=filter_lowpass,
                                     hp_filter=filter_highpass,
                                     filter_window_size=window_size,
                                     absolute=absolute_transformation
                                     )
    step_2 = reconstruction_algorithm((number_of_pixels_x, number_of_pixels_y, number_of_pixels_z),
                                      (field_of_view_x, field_of_view_y, field_of_view_z),
                                      **recon_params)
    step_1.add_child(step_2)
    return step_1


def read_unmixing_preset(json_path: Union[str, dict], example_data: Optional[PAData], **kwargs):
    """
    Load and parse the unmixing preset from the specified JSON file or dictionary.

    Parameters
    ----------
    json_path : Union[str, dict]
        The path to the JSON file or a dictionary containing the unmixing preset.

    example_data : PAData or None
        An example dataset used to obtain settings such as wavelengths.

    Returns
    -------
    unmixing_preset : SpatialProcessingAlgorithm
        The parsed unmixing preset as a pipeline element.

    """
    # Load json
    if type(json_path) == str:
        with open(json_path) as json_file:
            settings = json.load(json_file)
    else:
        settings = json_path
    settings.update(kwargs)

    # Load wavelengths from the preset or use the wavelengths from the example data
    wavelengths = settings.get(UnmixingAttributeTags.UNMIXING_WAVELENGTHS, None)
    wavelength_indices = settings.get(UnmixingAttributeTags.WAVELENGTH_INDICES, None)
    wavelength_range = settings.get(UnmixingAttributeTags.WAVELENGTH_RANGE, None)

    if wavelengths is None and wavelength_indices is None and wavelength_range is None:
        # Use all wavelengths from the example data
        wavelengths = example_data.get_wavelengths()[:]
    elif wavelengths is None and wavelength_indices is None:
        # Use all wavelengths within the specified range
        wavelengths = example_data.get_wavelengths()[:]
        wavelengths = wavelengths[(wavelengths >= wavelength_range[0]) & (wavelengths <= wavelength_range[1])]
    elif wavelengths is None:
        # Use the specified indices from the example data
        wavelengths = example_data.get_wavelengths()[:][wavelength_indices]

    # Get other unmixing settings from the preset
    resolution_reduction_factor = settings[UnmixingAttributeTags.RESOLUTION_REDUCE]
    spectra_names = settings[UnmixingAttributeTags.SPECTRA]

    # Initialize the spectral unmixer pipeline
    from patato.unmixing.spectra import SPECTRA_NAMES
    spectra = [SPECTRA_NAMES[x]() for x in spectra_names]

    compute_so2 = settings[UnmixingAttributeTags.COMPUTE_SO2]

    suffix = settings[UnmixingAttributeTags.SUFFIX]

    pipeline = SpectralUnmixer(spectra, wavelengths, resolution_reduction_factor, algorithm_id=suffix)

    # If haemoglobin and oxyhaemoglobin are present in the spectra, add THb and SO2 calculations to the pipeline
    if UnmixingAttributeTags.HAEMOGLOBIN in spectra_names and UnmixingAttributeTags.OXYHAEMOGLOBIN in spectra_names:
        thb = THbCalculator(suffix)
        pipeline.add_child(thb)
        if compute_so2:
            so2 = SO2Calculator(suffix)
            pipeline.add_child(so2)
    return pipeline
