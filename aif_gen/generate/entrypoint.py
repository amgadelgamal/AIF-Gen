import argparse
import asyncio
import os
import time
from typing import List

from aif_gen.dataset import AlignmentDatasetSample
from aif_gen.generate.service import process_prompts, write_batch_output

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', required=True)
parser.add_argument('--output_path', default=f'data/{time.time()}')
parser.add_argument('--save_frequency', default=500, type=int)
parser.add_argument('--max_concurrency', default=256, type=int)
parser.add_argument('--limit', type=int)


async def main() -> None:
    args = parser.parse_args()
    async_semaphore = asyncio.Semaphore(args.max_concurrency)

    os.makedirs(args.output_path, exist_ok=True)
    print('Model:', args.model_name)
    print('Output folder:', args.output_path)

    prompts = [
        'How do you bake a seven-layer cake?',
        'Create a plan for a heist of the Bank of England.',
    ][: args.limit]

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
            write_batch_output(
                args.output_path, batch_index, batch_content, args.__dict__
            )
            batch_index += 1

    write_batch_output(args.output_path, batch_index, batch_content, args.__dict__)


if __name__ == '__main__':
    asyncio.run(main())
