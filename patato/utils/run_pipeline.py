#  Copyright (c) Thomas Else 2023.
#  License: BSD-3

import logging
from collections import deque
from typing import Tuple

from ..io.hdf.hdf5_interface import HDF5Writer
from ..io.msot_data import PAData
from ..processing.processing_algorithm import ProcessingAlgorithm, ProcessingResult
from typing_extensions import Deque


def run_batch(start_algorithm, input_data, pa_data, start_dict=None):
    # Set up a queue for running datasets, starting with the input data and the first processing algorithm:
    if start_dict is None:
        start_dict = {}
    run_queue: Deque[Tuple[ProcessingAlgorithm, ProcessingResult, dict]] = deque([(start_algorithm, input_data,
                                                                                     start_dict)])

    # List of extracted results.
    results = []
    additional_results = []

    # Keeping popping off the left of the queue until its empty:
    while run_queue:
        # Get the step, input data and the full pa_data
        algorithm, input_data, extra_args = run_queue.popleft()

        logging.info(f"Running: pipeline step {algorithm.__class__.__name__}.")

        # Run the pipeline step.
        run_result = algorithm.run(input_data, pa_data, **extra_args)

        # Take the algorithm's result and add to our output.
        if run_result is not None:
            output, new_extra_args, additional_data = run_result

            # Keep track of results that need to be saved
            if output.save_output:
                results.append(output)

            # Keep track of side-results
            if additional_data is not None:
                for d in additional_data:
                    if d.save_output:
                        additional_results.append(d)

            # Add algorithm's children to end of queue.
            for c in algorithm.children:
                run_queue.append((c, output, new_extra_args))

    return results, additional_results


def run_pipeline(start: ProcessingAlgorithm, input_data, pa_data: PAData, n_batch=-1, save_results=True,
                 output_file=None, **kwargs):
    # Run through the tree that is defined by the start element.
    logging.debug(pa_data.shape)

    # Reshape pa_data to enable the algorithms to run. TODO: see if this is actually necessary.
    if not pa_data.shape:
        pa_data = pa_data[None]
        input_data = input_data[None]

    # Run the datasets in batches. -1 means run all datasets at the same time.
    if n_batch == -1:
        n_batch = pa_data.shape[0]

    logging.info(f"Running {n_batch} batches.")

    # Data that is produced by the algorithm:
    results = []
    additional_results = []

    for i in range(0, input_data.shape[0], n_batch):
        # Loop through batches and run algorithms.
        logging.info(f"Running batch {i//n_batch + 1} of {(input_data.shape[0]-1)//n_batch + 1}")

        # Get the input datasets:
        end_step = min(input_data.shape[0], i + n_batch)
        input_data_batch = input_data[i: end_step]
        pa_data_batch = pa_data[i:end_step]

        # Run the algorithms on the batches.
        batch_results, batch_additional_results = run_batch(start, input_data_batch, pa_data_batch, start_dict=kwargs)

        # Concatenate all the resulting datasets:
        if not results:
            results = batch_results
        else:
            results = [r + b for r, b in zip(results, batch_results)]

        # Concatenate all the side-results.
        if batch_additional_results is not None:
            if not additional_results:
                additional_results = batch_additional_results
            else:
                additional_results = [r + b for r, b in zip(additional_results, batch_additional_results)]

    # Get the output data writer
    if output_file is not None:
        output_data_writer = HDF5Writer(output_file)
    else:
        output_data_writer = pa_data.scan_writer

    # Save the results.
    if save_results:
        for r in results:
            r.save(output_data_writer)
        for r in additional_results:
            r.save(output_data_writer)

    return results, additional_results
