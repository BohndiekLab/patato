#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import glob
import xml.dom.minidom
from os.path import split, join

import numpy as np
from ...core.image_structures.reconstruction_image import Reconstruction
from ..hdf.fileimporter import ReaderInterface
from ..ithera import load_ithera_irf


def xml_to_dict(x):
    if x.nodeType == 3:
        return x.nodeValue
    elif not x.childNodes:
        return x.nodeName, x.nodeValue
    else:
        dicts = [xml_to_dict(y) for y in x.childNodes]
        v = [y for y in dicts if type(y) not in [str, list] or y[0] != "\n"]
        # convert to floats/ints:
        v = [y if type(y) is not str else int(y) if y.isnumeric() else float(y) if y.replace(".",
                                                                                             "").isnumeric() else y == "true" if y in [
            "true", "false"] else y for y in v]
        if type(v[0]) == tuple and all([y == v[0][0] for y, _ in v]):
            v = [y for _, y in v]
        elif type(v[0]) == tuple:
            v = dict(v)
        if type(v) == list and len(v) == 1:
            v = v[0]
        return x.nodeName, v


class iTheraMSOT(ReaderInterface):
    """An interface for iThera MSOT datasets.
    """
    def _get_rois(self):
        # In future, can extend this to enable import of iThera ROIS.
        pass

    def _get_segmentation(self):
        return None

    def _get_datasets(self):
        # Only support reconstructions for now. I think this could be extended easily enough to support
        # unmixed etc though
        # could easily get an error here in older versions of msot data format, so will try except - might need to check
        # that this at least doesn't fail on old versions...
        try:
            f = xml_to_dict(self.xml_tree.getElementsByTagName("ReconNodes")[0])
        except IndexError:
            return {}
        reconstructions = []
        rec_list = f[1] if type(f[1]) == list else [f[1]]

        for r in rec_list:
            if r is None:
                continue
            guid = r["GUID"]
            file = join(self.scan_folder, "RECONs", guid + ".bin")
            ns = [r["FIELD-OF-VIEW"]["PixelCount"][a] for a in "XYZ"]
            recon = np.memmap(file, dtype=np.single)[:self.nframes * self.nwavelengths * ns[0] * ns[1] * ns[2]].reshape(
                (self.nframes,
                 self.nwavelengths,) + tuple(ns))
            fov = [r["FIELD-OF-VIEW"]["Extents"][a] for a in "XYZ"]

            attributes = {x: r[x] for x in r.keys() if x != "ReconFrames"}

            # fill in required PATATO attributes:
            pat_attributes = {}
            pat_attributes["RECONSTRUCTION_ALGORITHM"] = "iThera " + attributes["Name"]
            pat_attributes["RECONSTRUCTION_NX"] = ns[0]
            pat_attributes["RECONSTRUCTION_NY"] = ns[1]
            pat_attributes["RECONSTRUCTION_NZ"] = ns[2]
            pat_attributes["RECONSTRUCTION_FIELD_OF_VIEW_NX"] = fov[0]
            pat_attributes["RECONSTRUCTION_FIELD_OF_VIEW_NY"] = fov[1]
            pat_attributes["RECONSTRUCTION_FIELD_OF_VIEW_NZ"] = fov[2]
            pat_attributes["RECONSTRUCTION_PARAMS"] = attributes
            pat_attributes["PREPROCESSING_ALGORITHM"] = "iThera " + attributes["SingalFilterType"]
            pat_attributes["speedofsound"] = attributes["TrimSpeedOfSound"]
            pat_attributes["Notes"] = "iThera Reconstruction, imported by PATATO. Speed of sound is offset from water " \
                                      "sos at the given temperature. "

            rec = Reconstruction(recon, self._get_wavelengths(),
                                 attributes=pat_attributes,
                                 field_of_view=fov, hdf5_sub_name="iThera " + attributes["Name"])
            reconstructions.append(rec)
        rec_dict = {}
        n_rec = {}
        for r in reconstructions:
            r_name = r.attributes["RECONSTRUCTION_ALGORITHM"]
            if r_name not in rec_dict:
                n_rec[r_name] = 0
            i = n_rec[r_name]
            n_rec[r_name] += 1
            rec_dict[(r_name, str(i))] = r
        if rec_dict:
            return {"recons": rec_dict}
        else:
            return {}

    def get_speed_of_sound(self):
        return None

    def __init__(self, folder, scan_name=None):
        super().__init__()
        if scan_name is None:
            scan_name = split(folder)[-1]
        self.scan_folder = folder
        self.xml_file = join(folder, scan_name + ".msot")
        self.xml_tree = xml.dom.minidom.parse(self.xml_file)
        # Start by establishing the number of frames. This can sometimes not be exactly what is expected
        # e.g. if you opened the scanner before it had finished one sweep.
        self._ithera_get_wavelengths()
        self.nwavelengths = len(self.wavelengths)

        v24_irf = self.xml_tree.getElementsByTagName("ImpulseResponse")
        if len(v24_irf) != 0 and len(v24_irf[0].firstChild.nodeValue) == 21656:
            import base64
            self.v24_irf = np.frombuffer(base64.b64decode(v24_irf[0].firstChild.nodeValue), dtype=np.double)
        else:
            self.v24_irf = None

        self.nframes = len(self.xml_tree.getElementsByTagName("DataModelScanFrame")) // self.nwavelengths
        # Optional add here: extract the reconstructed images that are from ViewMSOT.
        # Extract attributes
        self.scan_attrs = self._get_scan_attributes()
        self.scan_elements = self._get_scan_elements()
        self.nprojections = None
        self.geometry = self.get_sensor_geometry()
        self.nsamples = self.get_n_samples()
        self.water_absorption = self.get_water_absorption()

    def get_n_samples(self):
        N_samples = int(self.xml_tree.getElementsByTagName("RECORDED-LENGTH")[0].firstChild.nodeValue)
        return N_samples

    def _get_scan_attributes(self, attrs=None):
        if attrs is None:
            attrs = [("timestamp", np.uint64), ("ultraSound-frame-offset", int)]
        output = {}
        frames = self.xml_tree.getElementsByTagName("ScanFrames")[0].getElementsByTagName("Frame")
        for attribute_tag, attribute_type in attrs:
            output[attribute_tag] = np.zeros(self.nwavelengths * self.nframes, dtype=attribute_type)
            for frame_idx, frame in enumerate(frames[:self.nwavelengths * self.nframes]):
                output[attribute_tag][frame_idx] = attribute_type(frame.getAttribute(attribute_tag))

        for attribute_tag in output:
            output[attribute_tag] = output[attribute_tag].reshape((self.nframes, self.nwavelengths))

        return output

    def _get_scan_elements(self, elements=None):
        if elements is None:
            elements = [("TEMPERATURE", float), ("POWER", float), ("DIODE-READOUT", float),
                        ("LASER-ENERGY", float), ("Z-POS", float), ("RUN", int),
                        ("REPETITION", int), ("OverallCorrectionFactor", float),
                        ]
        frames = self.xml_tree.getElementsByTagName("ScanFrames")[0].getElementsByTagName("Frame")
        output = {}
        for element_tag, element_type in elements:
            output[element_tag] = np.zeros(self.nwavelengths * self.nframes, dtype=element_type)
            for frame_idx, frame in enumerate(frames):
                if frame_idx >= self.nwavelengths * self.nframes:
                    # This means that the final frame only has some of the wavelength measurements.
                    # We will cut off this last frame.
                    break
                elements_tag = frame.getElementsByTagName(element_tag)
                if len(elements_tag) > 0:
                    value = frame.getElementsByTagName(element_tag)[0].firstChild.nodeValue
                    value = element_type(value)
                    output[element_tag][frame_idx] = value
                else:
                    if element_type == int:
                        output[element_tag][frame_idx] = -1
                    else:
                        output[element_tag][frame_idx] = np.nan

        for element_tag in output:
            output[element_tag] = output[element_tag].reshape((self.nframes, self.nwavelengths))
        return output

    def _ithera_get_wavelengths(self):
        wavelength_tree = self.xml_tree.getElementsByTagName("WAVELENGTHS")[0]
        wavelength_lookup = []
        for wl_entry in wavelength_tree.getElementsByTagName("WAVELENGTH"):
            wavelength_lookup.append(float(wl_entry.firstChild.nodeValue))
        self.wavelengths = np.array(wavelength_lookup)

    def _get_wavelengths(self):
        return self.wavelengths

    def _get_correction_factor(self):
        if not np.all(np.isnan(self.scan_elements["POWER"][()])):
            return self.scan_elements["POWER"]
        else:
            return self.scan_elements["OverallCorrectionFactor"]

    def get_impulse_response(self):
        irf_files = glob.glob(join(self.scan_folder, "*.irf"))
        if len(irf_files) > 0:
            irf_file = irf_files[0]
            return load_ithera_irf(irf_file)
        else:
            return self.v24_irf

    def _get_repetition_numbers(self):
        return self.scan_elements["REPETITION"]

    def _get_run_numbers(self):
        return self.scan_elements["RUN"]

    def get_scan_datetime(self):
        """

        Returns
        -------

        """
        import dateutil.parser
        return dateutil.parser.isoparse(
            self.xml_tree.getElementsByTagName("CreationTime")[0].firstChild.nodeValue).replace(tzinfo=None)

    def _get_scan_times(self):
        return self.scan_attrs["timestamp"]

    def _get_temperature(self):
        return self.scan_elements["TEMPERATURE"]

    def get_us_offsets(self):
        return self.scan_attrs["ultraSound-frame-offset"]

    def _get_pa_data(self):
        raw_file = glob.glob(join(self.scan_folder, "*.bin"))[0]
        raw_data = np.memmap(raw_file, mode="r", dtype=np.uint16)[
                   :self.nframes * self.nwavelengths * self.nprojections *
                    self.nsamples].reshape(self.nframes, self.nwavelengths,
                                           self.nprojections, self.nsamples)
        return raw_data, {"fs": self.get_sampling_frequency()}

    def _get_sampling_frequency(self):
        return 1e6 * float(self.xml_tree.getElementsByTagName("SAMPLING-FREQUENCY")[0].firstChild.nodeValue)

    def _get_sensor_geometry(self):
        geometry = []
        frame_desc = self.xml_tree.getElementsByTagName("FRAME-DESC")[0]
        projection_xml = frame_desc.getElementsByTagName("PROJECTION")
        for n, f in enumerate(projection_xml):
            geometry.append([])
            for i, v in enumerate(f.getElementsByTagName("VALUE")):
                geometry[-1].append(float(v.firstChild.nodeValue))
        self.nprojections = len(geometry)
        geometry = np.array(geometry)
        return geometry

    def _get_water_absorption(self):
        coeffs = []
        start = self.xml_tree.getElementsByTagName("WATER-ABSORPTION-COEFFICIENTS")[0]
        for f in start.getElementsByTagName("WATER-ABSORPTION-COEFFICIENT"):
            if f.firstChild is not None:
                coeffs.append(float(f.firstChild.nodeValue))
            else:
                coeffs.append(float(f.getAttribute("coefficient")))
        pathlength = float(self.xml_tree.getElementsByTagName("PATH-LENGTH-IN-WATER")[0].firstChild.nodeValue)
        return np.array(coeffs), pathlength

    def get_us_data(self):
        us_files = glob.glob(join(self.scan_folder, "*.us"))
        if len(us_files) > 0:
            us_nodes = self.xml_tree.getElementsByTagName("ULTRA-SOUND-FIELD-OF-VIEW")
            if len(us_nodes) > 0:
                us_node = us_nodes[0]
                N = us_node.getElementsByTagName("PixelCount")[0].getElementsByTagName("X")[0].firstChild.nodeValue
                us_pixels = int(N)
                us_extents = us_node.getElementsByTagName("Extents")
                fov = float(us_extents[0].getElementsByTagName("X")[0].firstChild.nodeValue)
            else:
                N = self.xml_tree.getElementsByTagName("ULTRA-SOUND-RESOLUTION")[0].firstChild.nodeValue
                us_pixels = int(N)
                fov = float(
                    self.xml_tree.getElementsByTagName("UltraSoundPixelSize")[0].firstChild.nodeValue) * us_pixels
            try:
                us_data = np.memmap(us_files[0], mode="r", dtype=np.float32).reshape((-1, us_pixels, us_pixels))
                us_data = np.swapaxes(us_data, -1, -2)[:, ::-1, :]
                return us_data, fov
            except ValueError:
                print("Unable to import ultrasound scans - did the scanner fail to acquire the images? SKIPPING US")
                return None
        else:
            return None

    def get_scan_name(self):
        return self.xml_tree.getElementsByTagName("ScanNode")[0].getElementsByTagName("Name")[0].firstChild.nodeValue

    def _get_scanner_z_position(self):
        return self.scan_elements["Z-POS"]

    def get_scan_comment(self):
        try:
            return self.xml_tree.getElementsByTagName("Comment")[0].firstChild.nodeValue
        except AttributeError:
            return ""
        except IndexError:
            return ""
