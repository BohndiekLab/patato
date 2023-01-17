#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Union

import numpy as np


# ALL UNITS IN INVERSE CENTIMETRES (some per mole where specified)

class Spectrum(ABC):
    @staticmethod
    @abstractmethod
    def get_name() -> Union[str, None]:
        pass

    @staticmethod
    @abstractmethod
    def get_spectrum(wavelengths: np.ndarray) -> np.ndarray:
        pass


class Haemoglobin(Spectrum):
    # In cm-1/M, divide by 187 for indicative mua (see prahl)
    @staticmethod
    def get_name() -> str:
        return "Hb"

    @staticmethod
    def get_spectrum(wavelengths: np.ndarray) -> np.ndarray:
        from pandas import read_table
        folder = os.path.dirname(os.path.realpath(__file__))
        prahl_spec = read_table(join(folder, "spectra_files", "prahl.txt"))
        return np.interp(wavelengths, prahl_spec["lambda"], prahl_spec["Hb"], np.nan, np.nan)


class OxyHaemoglobin(Spectrum):
    # In cm-1/M, divide by 187 for indicative mua (see prahl)
    @staticmethod
    def get_name() -> str:
        return "HbO2"

    @staticmethod
    def get_spectrum(wavelengths: np.ndarray) -> np.ndarray:
        from pandas import read_table
        folder = os.path.dirname(os.path.realpath(__file__))
        prahl_spec = read_table(join(folder, "spectra_files", "prahl.txt"))
        return np.interp(wavelengths, prahl_spec["lambda"], prahl_spec["Hb02"], np.nan, np.nan)


class IndocyanineGreen(Spectrum):
    @staticmethod
    def get_name() -> str:
        return "ICG"

    @staticmethod
    def get_spectrum(wavelengths: np.ndarray) -> np.ndarray:
        from pandas import read_csv
        folder = os.path.dirname(os.path.realpath(__file__))
        df = read_csv(join(folder, "spectra_files", "ICG.csv"), header=None, sep=";", decimal=",")
        df = df[df[0] < 890]
        return np.interp(wavelengths, df[0], df[1], np.nan, np.nan)


class Melanin(Spectrum):
    # NOTE: Multiply by a volume fraction (~0.01-0.4) to get a realistic epidermis absorption.
    @staticmethod
    def get_name() -> str:
        return "Melanin"

    @staticmethod
    def get_spectrum(wavelengths: np.ndarray) -> np.ndarray:
        if type(wavelengths) != np.ndarray:
            wavelengths = np.array(wavelengths)
        return 1.7e12 * wavelengths ** (-3.48)


class Lipids(Spectrum):
    # From https://omlc.org/spectra/fat/fat.txt
    @staticmethod
    def get_name() -> str:
        return "Lipid"

    @staticmethod
    def get_spectrum(wavelengths: np.ndarray) -> np.ndarray:
        from pandas import read_table
        # print(os.path.realpath(__file__))
        folder = os.path.dirname(os.path.realpath(__file__))
        df = read_table(join(folder, "spectra_files", "lipids.txt"), skiprows=4)
        return np.interp(wavelengths, df["nm"], df["mu_a(1/m)"] / 300, np.nan, np.nan)


class Water(Spectrum):
    @staticmethod
    def get_name() -> str:
        return "H2O"

    @staticmethod
    def get_spectrum(wavelengths: np.ndarray) -> np.ndarray:
        import pandas as pd
        folder = os.path.dirname(os.path.realpath(__file__))
        water_file = join(folder, "spectra_files", "water.txt")
        water_short = pd.read_table(water_file)
        water_file_long = join(folder, "spectra_files", "water_long.txt")
        water_long = pd.read_table(water_file_long)

        water_long = water_long.sort_values("lambda", ignore_index=True)
        # regrid
        min_wavelength = water_short["lambda"].min()
        max_wavelength = water_long["lambda"].max()
        grid_wavelengths = np.arange(min_wavelength, max_wavelength, 5)
        short = np.interp(grid_wavelengths, water_short["lambda"], water_short["absorption"], np.nan, np.nan)
        long = np.interp(grid_wavelengths, water_long["lambda"], water_long["absorption"], np.nan, np.nan)
        absorption_values = np.nanmean(np.array([short, long]), axis=0)
        return np.interp(wavelengths, grid_wavelengths, absorption_values)


SPECTRA = [Haemoglobin, OxyHaemoglobin, IndocyanineGreen, Melanin, Water, Lipids]
SPECTRA_NAMES = {s.get_name(): s for s in SPECTRA}
