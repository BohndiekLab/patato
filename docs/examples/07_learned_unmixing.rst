Learned Spectral Unmixing
================

There exists a command line tool that runs an oxygenation estimation algorithm based on a machine learning model.
You have to specify a training dataset that the algorithm should be trained on.

.. warning::
   :caption: Experimental Algorithm

      The learned unmixing results have been shown to vary drastically with the training data set.
      Please make sure to chose a representative training dataset for your problem.

Running the algorithm
+++++++++++++

You can use the command line to run the learned unmixing algorithm on your data:

.. code-block:: bash
   :caption: Run learned unmixing with

        patato-learned_sO2 "path/to/HDF5_files/" -t TRAINING_DATA_NAME


Training data
^^^^^^^^^^^^^^^^^^^^^^
The following training data are available for unmixing. More details and ways to objectively chose a
suitable dataset will follow soon:

ACOUS
BASE
BG_0-100
BG-60-80
BG_H2O
HET_0-100
HET-60-80
ILLUM_5mm
ILLUM_POINT
INVIS_ACOUS
INVIS
INVIS_SKIN
INVIS_SKIN_ACOUS
MSOT_ACOUS
MSOT
MSOT_SKIN
MSOT_SKIN_ACOUS
RES_0.6
RES_0.15
RES_0.15_SMALL
RES_1.2
SKIN
SMALL
WATER_2cm
WATER_4cm


Reading the results
+++++++++++++

From Python, you can access the results using PATATO:

.. code-block:: Python
   :caption: Read the results

         import patato as pat
         import matplotlib.pyplot as plt

         # Load the photoacoustic data
         pa_data = pat.PAData.from_hdf5(r"H:\learned spectral unmixing\test_patato_learned_sO2/Scan_1.hdf5")

         # Look at all available keys
         print("learned unmixings:", list(pa_data.get_scan_learned_sO2().keys()))

         # Obtain unmixing results and the corresponding training data set from PATATO
         sO2_1 = pa_data.get_scan_learned_sO2()[('OpenCL Backprojection', '0learned_sO2')]
         sO2_1_dataset = sO2_1.attributes["TrainingDataset"]
         sO2_1_values = sO2_1.raw_data
