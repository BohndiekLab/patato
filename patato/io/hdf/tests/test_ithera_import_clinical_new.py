#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import unittest

from ...ithera.read_ithera import iTheraMSOT
from ...msot_data import PAData


class iTheraImportTestNew(unittest.TestCase):
    def test_import(self):
        p = PAData(iTheraMSOT(r"I:\research\seblab\data\group_folders\Tom\Datasets\Melanin_Acuity\raw_data\Scan_1"))
        p.set_default_recon()
        r = p.get_scan_reconstructions()


if __name__ == "__main__":
    itt = iTheraImportTestNew()
    itt.test_import()
