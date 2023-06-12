#  Copyright (c) Thomas Else 2023.
#  License: MIT

from os.path import isfile

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

from coverage import Coverage

cov = Coverage(source=['../patato'], omit=["*test*", "*/convenience_scripts/*", "*/useful_utilities/*"])
cov.start()

# noinspection PyPep8
from test_preprocessing_algorithm import TestPreprocessing
from test_backprojection import BackprojectionTest
from test_unmixing import TestUnmixing
from test_pipelines import TestPipelines
from test_image_sequence import TestHDF5Load
from test_ithera import TestITheraImport
from test_reconstruction_reading import TestJSONLoading
from test_msot_data import TestMSOTData
from test_model_based import TestModelBased
from make_dummy_dataset import make_dummy_dataset

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

make_dummy_dataset()

alltests = unittest.TestSuite()
alltests.addTest(unittest.makeSuite(TestMSOTData))
alltests.addTest(unittest.makeSuite(TestPreprocessing))
alltests.addTest(unittest.makeSuite(TestITheraImport))
alltests.addTest(unittest.makeSuite(BackprojectionTest))
alltests.addTest(unittest.makeSuite(TestUnmixing))
alltests.addTest(unittest.makeSuite(TestPipelines))
alltests.addTest(unittest.makeSuite(TestHDF5Load))
alltests.addTest(unittest.makeSuite(TestJSONLoading))
alltests.addTest(unittest.makeSuite(TestModelBased))

result = unittest.TextTestRunner(verbosity=2).run(alltests)

cov.stop()
cov.save()

cov.report(skip_empty=True, skip_covered=False)
cov.html_report(directory="../docs/test_coverage")
cov.xml_report(outfile="../docs/test_coverage/coverage.xml")
