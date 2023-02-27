import matplotlib.pyplot as plt
from patato.io.ipasc.read_ipasc import IPASCInterface
from patato import DefaultMSOTPreProcessor, ReferenceBackprojection, PAData


IPASC_data_file = r"H:\zurich_phantom_ring\ipasc\signal_P_5_1.hdf5"
SOUND_SPEED = 1488
pa_data = PAData(IPASCInterface(IPASC_data_file))

time_factor = 3
detector_factor = 2

preproc = DefaultMSOTPreProcessor(time_factor=time_factor, detector_factor=detector_factor,
                                  hilbert=True, lp_filter=7e6, hp_filter=5e3,
                                  irf=False)
recon = ReferenceBackprojection(field_of_view=(0.032, 0.032, 0.032), n_pixels=(300, 1, 300))

new_t1, d1, _ = preproc.run(pa_data.get_time_series(), pa_data)
rec1, _, _ = recon.run(new_t1, pa_data, SOUND_SPEED, **d1)
rec1 = rec1.raw_data

plt.imshow(rec1[0, 0, :, 0, :])
plt.show()
plt.close()