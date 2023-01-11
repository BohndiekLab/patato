File Format Specification
=============================

.. note::
    Note that it is not necessary to directly access the raw HDF5 files to use this library in Python. This guide is
    provided to enable access in other programming languages or for debugging purposes.

PATATO currently uses a custom HDF5 format to store the data. The structure of the file is as follows.

The root group contains the following attributes:

* `version` - The version of PATATO that created the file.
* `date` - The date and time the file was created.
* `comment` - A comment about the file.
* `fs` - The sampling frequency of the data.
* `name` - The name of the scan.
* `speedofsound` - The speed of sound of the scan if it has been set.

The root group contains the following datasets:

* `GEOMETRY` - (Required) The detection geometry of the scan. This is a dataset with shape (n detectors, 3), with the x, y, z positions of the ultrasound detectors.
* `raw_data` - (Required) The time domain data of the scan. This is a dataset with shape (n frames, n wavelengths, n detectors, n samples), with the raw data of the scan.
* `wavelengths` - (Required) The wavelengths used in the scan. Shape: (n wavelengths, ).
* `OverallCorrectionFactor` - (Optional) The laser energy correction factor for each run. This is a dataset with shape (n frames, n wavelengths).
* `REPETITION` - (Optional), the repetition number of each scan.
* `RUN` - (Optional) the run number of each scan.
* `TEMPERATURE` - (Optional) the temperature of the sample for each scan.
* `Z-POS` - (Optional) the z-position of the detectors for each scan (if the detectors are moved during the scan).
* `irf` - (Optional) the time-domain impulse response function for the detectors. Shape: (n samples, ).
* `timestamp` - (Optional) the timestamp of each scan. Shape: (n frames, n wavelengths).

Additional dataset groups are used for results generated in PATATO. Each of them have sub-groups specified as follows:

Each group has a subgroup with the reconstruction method e.g. "Reference Backprojection". In each of these sub-groups,
there are datasets numbered from 0 to n, so that multiple reconstructions can be stored in the same file.

E.g. ``file["recons/Reference Backprojection/0"]`` is a typical way of accessing the reconstructed data directly from
the HDF5 file in Python.

* `recons` - The reconstructed images. Each dataset has shape (n frames, n wavelengths, n x, n y, nz).
* `unmixed` - The unmixed images. Each dataset has shape (n frames, n unmixed, n x, n y, nz).
* `so2` - The unmixed sO2 (blood oxygenation) result. Each dataset has shape (n frames, 1, n x, n y, n z).
* `thb` - The unmixed THb (total haemoglobin) result. Each dataset has shape (n frames, 1, n x, n y, n z).

Their attributes give details about the parameters used to generate each result.

Regions of interest are stored as a list of vertices in the `roi` group, with subgroups named `name_position` and
numbered datasets. Each dataset has shape (n vertices, 2).
ROI dataset attributes contain information about the slice on which the roi was drawn and the name of the roi.

There is currently limited support for ultrasound images in PATATO, but they can be stored in the file. These are stored
in the `ultrasound` group. The offset between the ultrasound images and photoacoustic images is stored in the
`ultraSound-frame-offset` group.

There is also limited support for the couplant (e.g. water) absorption coefficients, but they can be stored in
`water-absorption-coefficients` dataset, with the path length stored in the attributes.
