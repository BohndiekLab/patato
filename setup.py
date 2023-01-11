#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import pybind11
import setuptools
from setuptools import Extension

requirements = [
    "requests",
    "cython",
    "shapely",
    "pylops",
    "h5py",
    "typing_extensions",
    "seaborn",
    "pandas",
    "scikit-learn",
    "scikit-fda",
    "xarray",
    "numpy",
    "scipy",
    "tabulate",
    "matplotlib",
    "pyopencl",
    "matplotlib_scalebar",
    "dask",
    "pybind11",
    "pydata-sphinx-theme",
    "jax",
    "simpa"
]

modelbased_c = Extension("patato.recon.model_based.generate_model",
                         sources=["patato/recon/model_based/generate_model.cpp"],
                         language="c++", extra_compile_args=["-std=c++11"],
                         include_dirs=[pybind11.get_include()], )
modelbased_c2 = Extension("patato.recon.model_based.generate_model_refraction",
                          sources=["patato/recon/model_based/generate_model_refraction.cpp"],
                          language="c++", extra_compile_args=["-std=c++11"],
                          include_dirs=[pybind11.get_include()], )

setuptools.setup(
    # Metadata
    name="patato",
    version="0.0.1",

    # Options
    packages=setuptools.find_namespace_packages(
        include=["patato", "patato.*"]
    ),
    install_requires=requirements,
    package_data={"patato.processing.spectra.spectra_files": ["*.csv", "*.txt"]},
    entry_points={"console_scripts": [
        "msot-import-ithera = patato.convenience_scripts.convert_binary_to_hdf5:main",
        "msot-draw-roi = patato.convenience_scripts.draw_roi:main",
        "msot-set-speed-of-sound = patato.convenience_scripts.tune_speed_of_sound:main",
        "msot-copy-roi = patato.convenience_scripts.copy_rois:main",
        "msot-generate-masks = patato.convenience_scripts.generate_masks:main",
        "msot-import-clinical = patato.convenience_scripts.import_clinical_data:main",
        "msot-import-ithera-recons = patato.convenience_scripts.import_ithera_recons:main",
        "msot-print-speeds = patato.convenience_scripts.print_speeds:main",
        "msot-rename-scans = patato.convenience_scripts.rename_scan:main",
        "msot-scan-status = patato.convenience_scripts.scan_status:main",
        "msot-unmix = patato.convenience_scripts.unmix:main",
        "msot-view = patato.convenience_scripts.view:main",
        "msot-reconstruct = patato.convenience_scripts.process_msot:main",
        "msot-analyse-dce = patato.convenience_scripts.data_analysis.analyse_dce:main",
        "msot-analyse-gc = patato.convenience_scripts.data_analysis.analyse_gas_challenge:main",
        "msot-split-rois = patato.convenience_scripts.split_rois:main",
        "msot-generate-core = patato.convenience_scripts.generate_core:main",
        "msot-convert-simpa = patato.convenience_scripts.convert_simpa:main"
    ]},
    ext_modules=[modelbased_c2, modelbased_c]
)
