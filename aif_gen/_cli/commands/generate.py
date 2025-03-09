import asyncio
import copy
import json
import logging
import pathlib
from typing import Optional

import click
import openai
import yaml
from huggingface_hub import HfApi

from aif_gen.generate.service import generate_continual_dataset
from aif_gen.util.path import get_run_id


@click.command(context_settings={'show_default': True})
@click.argument(
    'data_config_name',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    'model',
    type=click.STRING,
)
@click.option(
    '--output_file',
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    help='Path to write the generated dataset.',
    default=lambda: f'data/{get_run_id(name=click.get_current_context().params["data_config_name"].stem)}/data.json',
)
@click.option(
    '--max_concurrency',
    type=click.IntRange(min=1, max=1024, clamp=True),
    help='Max number of concurrent inference requests to send to the vLLM model',
    default=256,
)
@click.option(
    '-n',
    '--dry-run',
    is_flag=True,
    default=False,
    help='Ignore the input config and generate a dummy sample ensuring the model endpoint is setup.',
)
@click.option(
    '--hf-repo-id',
    type=str,
    default=None,
    help='If not None, push the generated dataset to a HuggingFace remote repository with the associated repo-id.',
)
def generate(
    data_config_name: pathlib.Path,
    model: str,
    output_file: pathlib.Path,
    max_concurrency: int,
    dry_run: bool,
    hf_repo_id: Optional[str],
) -> None:
    r"""Generate a new ContinualAlignmentDataset.

    DATA_CONFIG_NAME: Path to the dataset configuration file to use for dataset generation.
    MODEL: vLLM-compatible model to use for data generation.
    """
    logging.info(f'Using data configuration file: {data_config_name}')
    logging.info(f'Using model: {model}')

    data_config = yaml.safe_load(data_config_name.read_text())
    logging.debug(f'Configuration: {data_config}')

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        config = copy.deepcopy(data_config)
        config['model'] = model
        config['max_concurrency'] = max_concurrency
        with open(output_file.parent / 'config.json', 'w') as f:
            json.dump(config, f)

    try:
        client = openai.AsyncOpenAI()
    except (openai.OpenAIError, Exception) as e:
        logging.exception(f'Could not create openAI client: {e}')
        return

    async_semaphore = asyncio.Semaphore(max_concurrency)
    future = generate_continual_dataset(
        data_config, model, client, async_semaphore, dry_run
    )
    dataset = asyncio.get_event_loop().run_until_complete(future)
    if dataset is not None:
        logging.info(f'Writing {len(dataset)} samples to {output_file}')
        dataset.to_json(output_file)
        logging.info(f'Wrote {len(dataset)} samples to {output_file}')

        if hf_repo_id is not None:
            logging.info('Pushing dataset to HuggingFace')
            api = HfApi()
            api.create_repo(hf_repo_id, exist_ok=True, repo_type='dataset')
            api.upload_folder(
                folder_path=output_file.parent,
                repo_id=hf_repo_id,
                repo_type='dataset',
                allow_patterns='*.json',
            )
            logging.info('Pushed dataset to HuggingFace')
