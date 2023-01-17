"""
PATATO Reference
====================

All the functions, classes and modules contained in the PATATO package are detailed here.

For examples of how to use PATATO, please see the examples page.
"""

#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

from os import environ
from .io.ithera.read_ithera import iTheraMSOT
from .io.msot_data import PAData
from .io.simpa.read_simpa import SimpaImporter
from .recon.backprojection_opencl import OpenCLBackprojection
from .recon.backprojection_reference import ReferenceBackprojection

from .recon.model_based.model_based import ModelBasedReconstruction
from .recon.reconstruction_algorithm import ReconstructionAlgorithm

from .processing.jax_preprocessing_algorithm import MSOTPreProcessor

from .unmixing.unmixer import SpectralUnmixer, SO2Calculator, THbCalculator, GasChallengeAnalyser, DCEAnalyser
from .core.image_structures.reconstruction_image import Reconstruction
from .core.image_structures.unmixed_image import UnmixedData
from .core.image_structures.image_sequence import ImageSequence
from .utils.rois.roi_type import ROI
from .utils import sort_key
from .utils.run_pipeline import run_pipeline, run_batch
from .recon import get_default_recon_preset, get_default_unmixing_preset
from .io.json.json_reading import read_reconstruction_preset, read_unmixing_preset
from .core.image_structures.pa_time_data import PATimeSeries

Backprojection = ReferenceBackprojection

# For backwards compatibility:
DefaultMSOTPreProcessor = MSOTPreProcessor
GPUMSOTPreProcessor = MSOTPreProcessor

PAT_MAXIMUM_BATCH_SIZE = int(environ.get("PAT_MAXIMUM_BATCH_SIZE", 5))

""" DOCUMENTATION FIX: """
# To add support for importing to sphinx documentation:
__all_exports = [MSOTPreProcessor,
                 Backprojection, OpenCLBackprojection, ModelBasedReconstruction,
                 SpectralUnmixer, SO2Calculator, THbCalculator,
                 GasChallengeAnalyser, DCEAnalyser,
                 PAData,
                 SimpaImporter, iTheraMSOT,
                 Reconstruction, UnmixedData, ImageSequence,
                 ROI,
                 PATimeSeries]

for e in __all_exports:
    e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]
