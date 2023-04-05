Image Processing Parameters
==================================

The default image processing pipeline is as follows:

1. The dataset is converted from the original format into a PATATO HDF5 file (for example from the iThera format).

.. code-block:: bash

    patato-import-ithera /path/to/ithera/study/folder /path/to/data/folder

For the iThera importation, pass in the study folder, which contains Scan_1, Scan_2 etc. All of these will be processed,
producing Scan_1.hdf5, Scan_2.hdf5 etc in the data folder.

2. The speed of sound is set for each scan.

.. code-block:: bash

    patato-set-speed-of-sound /path/to/data/folder 1500

3. The image reconstruction algorithm is applied to each scan.

.. code-block:: bash

    patato-reconstruct /path/to/data/folder

4. Spectral unmixing can be applied.

.. code-block:: bash

    patato-unmix /path/to/data/folder

5. Regions of interest can be drawn over the images for further analysis.

.. code-block:: bash

    patato-draw-roi /path/to/data/folder

6. Time series analysis can be applied.

.. code-block:: bash

    patato-analyse-gc /path/to/data/folder --display True
    patato-analyse-dce /path/to/data/folder --display True

7. Further analysis can be done in Python (see examples).

.. code-block:: python

    >>> import patato as pat
    >>> dataset = pat.PAData('/path/to/data/folder/Scan_x.hdf5')
    >>> dataset.set_default_recon()
    >>> dataset.get_scan_reconstructions().imshow()

Reconstruction Preset Parameters
--------------------------------

Reconstruction parameters can be controlled by passing in `--preset`. The default preset is a good starting point for
most basic applications. It does a backprojection with a Hilbert transform. If a json file is passed in as a preset, you
can control the reconstruction parameters. For example, you can change the band-pass filter, remove the Hilbert
transform, change the pixel size, or use a Model-Based algorithm (implementation in progress).

.. code-block:: json
    :caption: The default reconstruction preset.

    {
        "FILTER_HIGH_PASS": 5e3, // High pass filter in Hz
        "FILTER_LOW_PASS": 7e6, // Low pass filter in Hz
        "IRF": true, // Whether to do impulse response correction
        "HILBERT_TRANSFORM": true, // Whether to do a Hilbert Transform
        "INTERPOLATE_TIME": 3, // Interpolate the time axis by this factor
        "INTERPOLATE_DETECTORS": 2, // Interpolate the detector axis by this factor
        "PREPROCESSING_ALGORITHM": "Standard Preprocessor", // Which preprocessing algorithm to use
        "RECONSTRUCTION_FIELD_OF_VIEW_X": 0.024975, // Field of view in x in metres
        "RECONSTRUCTION_FIELD_OF_VIEW_Y": 0.024975, // Field of view in y in metres
        "RECONSTRUCTION_FIELD_OF_VIEW_Z": 0., // Field of view in z in metres - ignored when RECONSTRUCTION_NZ is 1
        "RECONSTRUCTION_NX": 333, // Number of pixels in x
        "RECONSTRUCTION_NY": 333, // Number of pixels in y
        "RECONSTRUCTION_NZ": 1, // Number of pixels in z
        "RECONSTRUCTION_PARAMS": {}, // Extra parameters for the reconstruction algorithm
        "RECONSTRUCTION_ALGORITHM": "Reference Backprojection" // Which reconstruction algorithm to use
    }

Unmixing Preset Parameters
---------------------------

Unmixing parameters can also be controlled by passing in `--preset`. By default, the unmixing is done with
Oxyhaemoglobin and Deoxyhaemoglobin basis spectra. To unmixing for different chromophores, pass in a json file as a
preset.

.. code-block:: json
    :emphasize-lines: 4
    :caption: The default unmixing preset.

    {
        "RESOLUTION_REDUCE": 3, // The factor by which to reduce the resolution of the reconstruction to improve SNR
        "WAVELENGTH_RANGE": [700, 900], // The wavelength range to use for unmixing
        "SPECTRA": ["Hb", "HbO2"], // The chromophores to use as basis for unmixing. Could also add "ICG".
        "SO2": true, // Whether to calculate sO2 after unmixing
        "SUFFIX": "" // What label to give the unmixing with this preset (e.g. ICG) - this allows you to make sure that
        // you use the correct unmixing in your analysis. It makes no difference to the actual algorithm.
    }
