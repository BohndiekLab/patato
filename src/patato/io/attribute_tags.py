#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

class PreprocessingAttributeTags:
    HIGH_PASS_FILTER = "FILTER_HIGH_PASS"
    LOW_PASS_FILTER = "FILTER_LOW_PASS"
    TIME_INTERPOLATION = "INTERPOLATE_TIME"
    DETECTOR_INTERPOLATION = "INTERPOLATE_DETECTORS"
    IMPULSE_RESPONSE = "IRF"
    HILBERT_TRANSFORM = "HILBERT_TRANSFORM"
    ENVELOPE_DETECTION = "ENVELOPE_DETECTION"
    WINDOW_SIZE = "WINDOW_SIZE"
    PROCESSING_ALGORITHM = "PREPROCESSING_ALGORITHM"
    COUPLANT_CORRECTION = "COUPLANT_CORRECTION"
    COUPLANT_PATH_LENGTH = "COUPLANT_PATH_LENGTH"


class ReconAttributeTags:
    ADDITIONAL_PARAMETERS = "RECONSTRUCTION_PARAMS"
    X_FIELD_OF_VIEW = "RECONSTRUCTION_FIELD_OF_VIEW_X"
    Y_FIELD_OF_VIEW = "RECONSTRUCTION_FIELD_OF_VIEW_Y"
    Z_FIELD_OF_VIEW = "RECONSTRUCTION_FIELD_OF_VIEW_Z"
    X_NUMBER_OF_PIXELS = "RECONSTRUCTION_NX"
    Y_NUMBER_OF_PIXELS = "RECONSTRUCTION_NY"
    Z_NUMBER_OF_PIXELS = "RECONSTRUCTION_NZ"
    RECONSTRUCTION_ALGORITHM = "RECONSTRUCTION_ALGORITHM"
    OLD_FIELD_OF_VIEW = "RECON_FOV"
    OLD_RECON_NX = "RECON_NX"


class UnmixingAttributeTags:
    RESOLUTION_REDUCE = "RESOLUTION_REDUCE"
    WAVELENGTH_RANGE = "WAVELENGTH_RANGE"
    WAVELENGTH_INDICES = "WAVELENGTH_INDICES"
    UNMIXING_WAVELENGTHS = "WAVELENGTHS"
    SPECTRA = "SPECTRA"
    COMPUTE_SO2 = "SO2"
    SUFFIX = "SUFFIX"
    HAEMOGLOBIN = "Hb"
    OXYHAEMOGLOBIN = "HbO2"
    MELANIN = "Melanin"
    ICG = "ICG"


class ROITags:
    Z_POSITION = "z"
    REPETITION = "repetition"
    ROI_NAME = "class"
    ROI_POSITION = "position"
    RUN = "run"
    GENERATED_ROI = "generated"


class GCAttributeTags:
    STEPS = "steps"
    BUFFER = "buffer"
    SKIP_START = "skip_start"


class HDF5Tags:
    POWER = "POWER"
    OVERALL_CORR = "OverallCorrectionFactor"
    RAW_DATA = "raw_data"
    RECONSTRUCTION = "recons"
    UNMIXED = "unmixed"
    SO2 = "so2"
    THB = "thb"
    SPEED_OF_SOUND = "speedofsound"
    SAMPLING_FREQ = "fs"
    SCAN_GEOMETRY = "GEOMETRY"
    WAVELENGTH = "wavelengths"
    IMPULSE_RESPONSE = "irf"
    Z_POSITION = "Z-POS"
    REPETITION = "REPETITION"
    REGIONS_OF_INTEREST = "rois"
    RUN = "RUN"
    DELTA_SO2 = "dso2"
    BASELINE_SO2 = "baseline_so2"
    BASELINE_SO2_STANDARD_DEVIATION = "baseline_so2_sigma"
    TIMESTAMP = "timestamp"
    TEMPERATURE = "TEMPERATURE"
    ULTRASOUND_FRAME_OFFSET = "ultraSound-frame-offset"
    DATE = "date"
    ORIGINAL_NAME = "original_name"
    SCAN_COMMENT = "comment"
    WATER_ABSORPTION_COEFF = "water-absorption-coefficients"
    WATER_PATHLENGTH = "pathlength"
    ULTRASOUND = "ultrasound"
    ULTRASOUND_FIELD_OF_VIEW = "fov"
    SCAN_NAME = "name"
    SEGMENTATION = "seg"
    DELTA_ICG = "dicg"
    BASELINE_ICG = "baseline_icg"
    BASELINE_ICG_SIGMA = "baseline_icg_sigma"
