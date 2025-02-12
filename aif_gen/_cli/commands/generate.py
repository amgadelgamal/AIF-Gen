import asyncio
import logging
import pathlib
import time

import click
import openai
import yaml


@click.command(context_settings={'show_default': True})
@click.argument(
    'data_config', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path)
)
@click.option(
    '--model',
    type=click.STRING,
    help='vLLM-compatible model to use for data generation',
    default='Meta-Llama-3.1-8B-Instruct',
)
@click.option(
    '--output_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    help='Path to write the generated dataset.',
    default=f'data/{time.time()}/data.json',
)
@click.option(
    '--max_concurrency',
    type=click.IntRange(min=1, max=1024, clamp=True),
    help='Max number of concurrent inference requests to send to the vLLM model',
    default=256,
)
def generate(
    data_config: pathlib.Path,
    model: str,
    output_file: pathlib.Path,
    max_concurrency: int,
) -> None:
    r"""Generate a new ContinualAlignmentDataset.

    DATA_CONFIG: Path to the dataset configuration file to use for dataset generation.
    """
    logging.info(f'Using data configuration file: {data_config}')
    logging.info(f'Using model: {model}')

    config_dict = yaml.safe_load(data_config.read_text())
    logging.debug(f'Configuration: {config_dict}')

    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        _ = openai.AsyncOpenAI()
    except (openai.OpenAIError, Exception) as e:
        logging.exception(f'Could not create openAI client: {e}')
        return

    _ = asyncio.Semaphore(max_concurrency)
    # TODO: Setup async click
    # dataset = await generate_continual_dataset(config_dict, client, async_semaphore)
    dataset = None
    if dataset is not None:
        logging.info(f'Writing {len(dataset)} samples to {output_file}')
        dataset.to_json(output_file)
        logging.info(f'Wrote {len(dataset)} samples to {output_file}')
