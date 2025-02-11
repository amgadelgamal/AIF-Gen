import argparse
import asyncio
import logging
import time

import openai
import yaml

from aif_gen.generate.service import generate_continual_dataset
from aif_gen.util.logging import setup_basic_logging
from aif_gen.util.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Generate an AlignmentDataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config_file',
    type=str,
    default='config/aif_config.yaml',
    help='Path to configuration file to use.',
)
parser.add_argument(
    '--log_file',
    type=str,
    default=f'aif_generation.log',
    help='Name of the log file to write to within the output directory.',
)
parser.add_argument(
    '--output_file',
    type=str,
    default=f'data/{time.time()}/data.json',
    help='Path to save the dataset.',
)
parser.add_argument(
    '--max_concurrency',
    type=int,
    default=256,
    help='Max number of concurrent API inference requests to the language model.',
)


async def main() -> None:
    args = parser.parse_args()

    log_file = get_root_dir() / args.log_file
    setup_basic_logging(log_file)

    config_file = get_root_dir() / args.config_file
    logging.info(f'Using configuration file: {config_file}')
    config_dict = yaml.safe_load(config_file.read_text())
    logging.debug(f'Configuration: {config_dict}')

    output_file = get_root_dir() / args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    async_semaphore = asyncio.Semaphore(args.max_concurrency)

    try:
        client = openai.AsyncOpenAI()
    except (openai.OpenAIError, Exception) as e:
        logging.exception(f'Could not create openAI client: {e}')
        return

    dataset = await generate_continual_dataset(config_dict, client, async_semaphore)
    if dataset is not None:
        logging.info(f'Writing {len(dataset)} samples to {output_file}')
        dataset.to_json(output_file)
        logging.info(f'Wrote {len(dataset)} samples to {output_file}')


if __name__ == '__main__':
    asyncio.run(main())
