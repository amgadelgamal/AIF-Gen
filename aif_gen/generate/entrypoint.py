import argparse
import asyncio
import logging
import pathlib
import time

import yaml

from aif_gen.api.prompt_mapper import PromptMapper
from aif_gen.api.response_mapper import ResponseMapper
from aif_gen.dataset import AlignmentDataset
from aif_gen.generate.service import process_prompts
from aif_gen.task import AlignmentTask
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
    '--save_frequency',
    type=int,
    default=500,
    help='Number of samples to batch write to disc.',
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

    async_semaphore = asyncio.Semaphore(args.max_concurrency)

    task = AlignmentTask.from_dict(config_dict['alignment_task'])
    logging.info(f'Generating AIF Dataset for task: {task}')
    logging.info(f'Using Model: {config_dict["model_name"]}')

    # TODO: Technically these need to be done sequentially
    task_prompt = get_task_prompt(task)
    response_prompt = get_response_prompt(task, task_prompt, config_dict['num_samples'])

    batch_index: int = 0
    dataset = AlignmentDataset(task, [])
    async for sample in process_prompts(
        prompts=[task_prompt, response_prompt],
        model_name=config_dict['model_name'],
        async_semaphore=async_semaphore,
    ):
        if sample is not None:
            dataset.append(sample)

        if (len(dataset) + 1) % args.save_frequency == 0:
            write_dataset_batch(dataset, output_path, batch_index)
            dataset = AlignmentDataset(task, [])  # Should be able to just clear it
            batch_index += 1

    output_file_path = output_path / f'output_{batch_index:03d}.json'
    dataset.to_json(output_file_path)


def write_dataset_batch(
    dataset: AlignmentDataset, output_path: pathlib.Path, batch_index: int
) -> None:
    output_file_path = output_path / f'output_{batch_index:03d}.json'
    logging.info(f'Writing batch with {len(dataset)} samples to {output_file_path}')
    dataset.to_json(output_file_path)
    logging.info(f'Wrote batch with {len(dataset)} samples to {output_file_path}')


def get_task_prompt(task: AlignmentTask) -> str:
    logging.info(f'Generating task prompt for {task}')

    prompt_mapper = PromptMapper()
    prompt = prompt_mapper.generate_prompt(task)
    logging.info(f'Using task prompt:\n\n{prompt}')
    return prompt


def get_response_prompt(task: AlignmentTask, task_prompt: str, num_samples: int) -> str:
    logging.info(f'Generating response prompt ({num_samples} samples)')

    response_mapper = ResponseMapper()
    prompt = response_mapper.generate_prompt(task, task_prompt, num_samples=num_samples)
    logging.info(f'Using response prompt:\n\n{prompt}')
    return prompt


if __name__ == '__main__':
    asyncio.run(main())
