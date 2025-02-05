import argparse
import asyncio
import pathlib
import time
from typing import List

from aif_gen.dataset import AlignmentDatasetSample
from aif_gen.generate.service import process_prompts, write_batch_output
from aif_gen.task import AlignmentTask, Domain

parser = argparse.ArgumentParser(
    description='Generate an AlignmentDataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--model_name', type=str, required=True, help='OpenAI compatible model alias.'
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
parser.add_argument(
    '--num_samples',
    type=int,
    default=1000,
    help='Number of response pairs to generate for th egiven task prompt.',
)


async def main() -> None:
    args = parser.parse_args()
    async_semaphore = asyncio.Semaphore(args.max_concurrency)

    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print('Model:', args.model_name)
    print('Output folder:', args.output_path)

    prompts = []
    prompts.append(generate_mock_prompt(args.num_samples))

    # Save occasionally.
    batch_index: int = 0

    # TODO: replace with actual Alignment dataset.
    batch_content: List[AlignmentDatasetSample] = []
    async for sample in process_prompts(
        prompts, model_name=args.model_name, async_semaphore=async_semaphore
    ):
        if sample is not None:
            batch_content.append(sample)

        if (len(batch_content) + 1) % args.save_frequency == 0:
            write_batch_output(output_path, batch_index, batch_content, args.__dict__)
            batch_index += 1

    write_batch_output(output_path, batch_index, batch_content, args.__dict__)


def generate_mock_prompt(num_samples: int) -> str:
    get_mock_alignment_task()
    return 'Mock'


def get_mock_alignment_task() -> AlignmentTask:
    component_dict = {
        'Healthcare': {
            'seed_words': ['hospital', 'exercise', 'medicine'],
        },
        'Technology': {
            'seed_words': ['engineering', 'AI', 'internet'],
        },
    }
    domain = Domain.from_dict(component_dict)
    objective = 'Generate a news article headline.'
    preference = 'Make the headline polarizing'

    task = AlignmentTask(domain, objective, preference)
    return task


if __name__ == '__main__':
    asyncio.run(main())
