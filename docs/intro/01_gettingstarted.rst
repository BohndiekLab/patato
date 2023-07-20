Installation
================

In order to use PATATO, you must have a Python environment set up on your computer. We recommend using
Anaconda (http://anaconda.com) to run Python, particularly if you are using Windows. You may wish to setup
a separate Anaconda environment to install PATATO, to minimise conflicts between dependency versions.

.. tip::
    If you are using Anaconda, you may wish to create a new environment before installing PATATO. This can be
    done by running the following command in the Anaconda prompt:

        conda create -n patato python=3.9

    Then activate the environment by running:

        conda activate patato

    You can then install PATATO as normal.

Installation
+++++++++++++

Option 1: Install using pip
------------------------------------------------------

Once you have Python installed, you can install PATATO using pip:

.. code-block:: bash
   :caption: Install PATATO using pip.

        pip install --upgrade pip setuptools
        pip install patato

To add GPU support, follow the guide on the JAX official guide here: (https://github.com/google/jax#installation).

Option 2: Install from source
------------------------------------

To install the most recent version of PATATO from GitHub:

    pip install git+https://github.com/tomelse/patato

Option 3: Install from source (editable)
----------------------------------------------------------

To install the development version of PATATO from GitHub and allow editing for development purposes:

.. code-block::
   :caption: Install PATATO from source.

        cd /path/to/installation/directory
        git clone https://github.com/tomelse/patato
        cd patato
        pip install -e .
