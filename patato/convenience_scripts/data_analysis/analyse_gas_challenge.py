#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import argparse
import glob
from os.path import join

from ...io.msot_data import PAData
from ...unmixing.unmixer import GasChallengeAnalyser
from ...utils import sort_key
from ...utils.run_pipeline import run_pipeline


def init_argparse():
    parser = argparse.ArgumentParser(description="Analyse Gas Challenge Data.")
    parser.add_argument('input', type=str, help="Data Folder")
    parser.add_argument('-p', '--prefix', type=str, help="Gas Challenge name prefix")
    parser.add_argument('-w', '--window', type=int, help="Smoothing window size",
                        default=10)
    parser.add_argument('-b', '--buffer', type=int, help="Buffer around changes",
                        default=5)
    parser.add_argument('-d', '--display', type=bool, help="Display steps",
                        default=True)
    parser.add_argument('-s', '--sigma', type=float, help="Smoothing window sigma",
                        default=4)
    parser.add_argument('-g', '--gas', type=lambda x: -1 if "air" == x else 1,
                        help="The starting gas for the challenge. If air then standard GC, "
                             "if o2 then it switches from o2->air", default="o2")
    parser.add_argument('--skipstart', type=int, help="Skip Runs at Start",
                        default=0)
    return parser


def main():
    p = init_argparse()
    args = p.parse_args()

    data_folder = args.input
    prefix = args.prefix

    if prefix is None:
        prefix = ""
        raise RuntimeWarning("Warning: You should usually set a prefix for this analysis.")

    for file in sorted(glob.glob(join(data_folder, "*.hdf5")), key=sort_key):
        data = PAData.from_hdf5(file, "r+")
        scan_name = data.get_scan_name().lower()
        prefices = prefix.lower().split(",")
        if not any([x in scan_name for x in prefices]):
            continue
        data.clear_dso2()
        print(scan_name, file)

        scan_so2s = data.get_scan_so2()
        if not scan_so2s:
            print("Skipped", file, "- no so2.")
            continue

        analyser = GasChallengeAnalyser(args.window, args.display, args.sigma,
                                        args.skipstart, args.gas, args.buffer)
        try:
            for recon in list(scan_so2s.keys()):
                so2_data = scan_so2s[recon]
                run_pipeline(analyser, so2_data, data, -1, True)
        except RuntimeError:
            print("---- SCAN NOT PROCESSED (no reference region) ---")
        data.file.close()


if __name__ == "__main__":
    main()
