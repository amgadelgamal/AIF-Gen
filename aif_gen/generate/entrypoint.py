import argparse
import asyncio
import logging
import time

import yaml

from aif_gen.generate.service import process_tasks
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
    '--output_path',
    type=str,
    default=f'data/{time.time()}',
    help='Path to directory where to save the dataset.',
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
    config_dict = yaml.safe_load(config_file.read_text())
    logging.info(f'Using configuration: {config_dict}')

    output_path = get_root_dir() / args.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / 'train.json'  # TODO: Waiting on merge #44

    async_semaphore = asyncio.Semaphore(args.max_concurrency)

    # TODO: Should directly construct a ContinualAlignmentDataset (or incrementally flush it to disc)
    async for dataset in process_tasks(config_dict, async_semaphore=async_semaphore):
        logging.info(f'Writing {len(dataset)} samples to {output_file_path}')
        dataset.to_json(output_file_path)
        logging.info(f'Wrote {len(dataset)} samples to {output_file_path}')


if __name__ == '__main__':
    asyncio.run(main())
