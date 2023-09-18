Running Tests
=====================================================================

Unit tests have been written for the majority of the functionality of PATATO. To rerun the tests,
run the following in the command line:

.. code-block:: console

    git clone https://github.com/BohndiekLab/patato
    cd patato
    pip install -e .
    cd tests
    python -m unittest

This will output ok for each successful test. Note some of the tests will not run without a GPU, decreasing the coverage.
