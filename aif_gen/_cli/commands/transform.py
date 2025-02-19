import logging
import pathlib

import click

import aif_gen.dataset.transforms.functional as F
from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
)


@click.group()
def transform() -> None:
    r"""Transform a ContinualAlignmentDataset."""


@transform.command(context_settings={'show_default': True})
@click.argument(
    'input_data_file',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    'output_data_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    '--p',
    type=click.FloatRange(min=0, max=1),
    default=1,
    help="Probability with which to swap each 'chosen' and 'rejected' in the dataset",
)
def preference_swap(
    input_data_file: pathlib.Path, output_data_file: pathlib.Path, p: float
) -> None:
    r"""Swap the 'chosen' and 'rejected' respones for each sample in the dataset with probability.

    INPUT_DATA_FILE: Path to the input dataset.
    OUTPUT_DATA_FILE: Path to the output (transformed) dataset.
    """
    logging.info(f'Reading dataset from: {input_data_file}')
    dataset = ContinualAlignmentDataset.from_json(input_data_file)
    logging.info(f'Read {len(dataset)} samples from: {input_data_file}')

    logging.info(f'Applying preference swap transform with p={p}')
    transformed_dataset = F.preference_swap_transform(dataset, swap_probability=p)
    logging.info(f'Transformed dataset.')

    logging.info(f'Writing dataset to: {output_data_file}')
    transformed_dataset.to_json(output_data_file)
    logging.info(f'Wrote {len(transformed_dataset)} samples from: {output_data_file}')
