import patato as pat
from patato.recon.fourier_transform_rec import FFTReconstruction
from patato.processing.numpy_basic_bandpassfilter import NumpyBasicPreProcessor
import matplotlib.pyplot as plt
import numpy as np

# Setup the classes for reconstruction and preprocessing
# N.b. my code checks that the supplied values are 2D and then ignores the third dimension. This is so the classes generalise to 3D.

# N.b. all SI units here.
ftr = FFTReconstruction(field_of_view=(0.04, 0, 0.04), n_pixels=(400, 1, 400))
# This applies a butterworth filter.
preproc = NumpyBasicPreProcessor(lp_filter=7e6, hp_filter=5e3)

pa = pat.PAData.from_hdf5(
    "/Users/else01/patato-dev/patato/docs/examples/misc/ExamplePhantom.hdf5"
)

pa_time_series, _, _ = preproc.run(
    pa.get_time_series(), pa
)  # Passing in the full pa data allows laser energy correction
recon, _, _ = ftr.run(pa_time_series, pa, speed_of_sound=1520)
recon.imshow()
plt.show()

### OPTION 2:
## OR, to do the above manually just with NumPy arrays:

timeseries = np.load(
    "/Users/else01/patato-dev/patato/docs/examples/misc/timeseries.npy"
).reshape(
    (1, 1, 256, 2030)
)  # PATATO deals with (nruns, nwavelengths, ndetectors, ntimesamples)
fs = 4e7  # Hz, sampling frequency
c = 1520  # m/s, speed of sound
detector_angle = 120  # Degrees
ndetectors = 256
detector_angles = (
    np.linspace(-detector_angle / 2, detector_angle / 2, ndetectors) * np.pi / 180
)
radius = 0.04  # m
detectors = np.array(
    [
        radius * np.cos(detector_angles),
        np.zeros_like(detector_angles),
        radius * np.sin(detector_angles),
    ]
).T  # Shape must be (ndetectors, 3)

# Wrap the time series data in a PATATO class
pa_timeseries = pat.PATimeSeries.from_numpy(timeseries, [700], fs, c)
#                           numpy array, light wavelength(irrelevant here), fs, speed of sound

# N.b. if you want to do a laser energy correction here, you have to divide through manually (I haven't bothered)

# Apply a butterworth filer
pa_timeseries, _, _ = preproc.run(pa_timeseries)

# Reconstruct
recon, _, _ = ftr.run(pa_timeseries, speed_of_sound=c, geometry=detectors)

recon.imshow()
plt.show()
