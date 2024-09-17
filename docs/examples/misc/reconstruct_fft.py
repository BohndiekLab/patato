import patato as pat
from patato.recon.fourier_transform_rec import FFTReconstruction

pa = pat.PAData.from_hdf5("/Volumes/Extreme SSD/Papers/PAISKINTONE/")
ftr = FFTReconstruction()
