import json
import logging
import pathlib
from typing import Any, Dict

import click

from aif_gen.dataset.continual_alignment_dataset import (
    ContinualAlignmentDataset,
)
from aif_gen.dataset.validation import (
    AlignmentEvaluator,
    CoherenceEvaluator,
    ContrastEvaluator,
    DiversityEvaluator,
    RelevanceEvaluator,
    count_validation,
    entropy_validation,
)


@click.command(context_settings={'show_default': True})
@click.argument(
    'input_data_file',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    'output_validation_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    '--validate-count/--no-validate-count',
    is_flag=True,
    default=True,
    help='Perform basic count validation on the dataset.',
)
@click.option(
    '--validate-entropy/--no-validate-entropy',
    is_flag=True,
    default=True,
    help='Perform entropy validation on the dataset.',
)
@click.option(
    '--validate-alignment/--no-validate-alignment',
    is_flag=True,
    default=True,
    help='Perform alignment validation on the dataset.',
)
@click.option(
    '--validate-coherence/--no-validate-coherence',
    is_flag=True,
    default=True,
    help='Perform coherence validation on the dataset.',
)
@click.option(
    '--validate-contrast/--no-validate-contrast',
    is_flag=True,
    default=True,
    help='Perform contrast validation on the dataset.',
)
@click.option(
    '--validate-diversity/--no-validate-diversity',
    is_flag=True,
    default=True,
    help='Perform diversity validation on the dataset.',
)
@click.option(
    '--validate-relevance/--no-validate-relevance',
    is_flag=True,
    default=True,
    help='Perform relevance validation on the dataset.',
)
def validate(
    input_data_file: pathlib.Path,
    output_validation_file: pathlib.Path,
    validate_count: bool,
    validate_entropy: bool,
    validate_alignment: bool,
    validate_coherence: bool,
    validate_contrast: bool,
    validate_diversity: bool,
    validate_relevance: bool,
) -> None:
    r"""Validate a ContinualAlignmentDataset.

    INPUT_DATA_FILE: Path to the input dataset.
    OUTPUT_VALIDATION_FILE: Path to the output validation file.
    """
    logging.info(f'Reading dataset from: {input_data_file}')
    dataset = ContinualAlignmentDataset.from_json(input_data_file)
    logging.info(f'Read {len(dataset)} samples from: {input_data_file}')

    results: Dict[str, Any] = {}
    if validate_count:
        logging.info('Performing count validation')
        results['count_validation'] = count_validation(dataset)
        logging.info('Finished count validation')

    if validate_entropy:
        logging.info('Performing entropy validation')
        results['entropy_validation'] = entropy_validation(dataset)
        logging.info('Finished entropy validation')

    if validate_alignment:
        logging.info('Performing alignment validation')
        results['alignment_validation'] = AlignmentEvaluator().evaluate_all(dataset)
        logging.info('Finished alignment validation')

    if validate_coherence:
        logging.info('Performing coherence validation')
        results['coherence_validation'] = CoherenceEvaluator().evaluate_all(dataset)
        logging.info('Finished coherence validation')

    if validate_contrast:
        logging.info('Performing contrast validation')
        results['contrast_validation'] = ContrastEvaluator().evaluate_all(dataset)
        logging.info('Finished contrast validation')

    if validate_diversity:
        logging.info('Performing diversity validation')
        results['diversity_validation'] = DiversityEvaluator().evaluate_all(dataset)
        logging.info('Finished diversity validation')

    if validate_relevance:
        logging.info('Performing relevance validation')
        results['relevance_validation'] = RelevanceEvaluator().evaluate_all(dataset)
        logging.info('Finished relevance validation')

    if len(results):
        logging.info(f'Writing validation results to: {output_validation_file}')
        with output_validation_file.open('w', encoding='utf-8') as f:
            json.dump(results, f)
    else:
        logging.warning('No validation measure was specified, skipping writedown.')
