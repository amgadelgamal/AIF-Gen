import asyncio
import json
import os
from typing import AsyncGenerator, Dict, List, Optional, Union

import backoff
import openai
import pydantic
from tqdm.asyncio import tqdm

from aif_gen.dataset import AlignmentDatasetSample


class _OutputModel(pydantic.BaseModel):
    """Synthetic preference based on the given principles."""

    chosen: str
    rejected: str


TEMPLATE = """\
Scenario: {scenario}

Principles:
- The response should be safe and promote ethical values.
- The response should be helpful, provided that the response is safe and ethical.

Given the scenario above, create a pair of response: "chosen", and "rejected", \
such that the "chosen" response follows the given "Principles" better than the "rejected" prompt.
"""  # TODO: replace with actual instructions

client = openai.AsyncOpenAI()


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def generate(
    prompt: str,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> Optional[AlignmentDatasetSample]:
    """ChatCompletion generation.

    Returns None if output is not parsed.
    """
    async with async_semaphore:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'SyntheticPreference',
                    'schema': _OutputModel.model_json_schema(),
                    'strict': True,
                },
            },
        )

    output = response.choices[0].message.content
    assert output is not None

    try:
        response = _OutputModel.model_validate_json(output)
        return AlignmentDatasetSample(
            prompt=prompt, chosen=response.chosen, rejected=response.rejected
        )
    except pydantic.ValidationError as e:
        print(e)
        return None


async def process_prompts(
    prompts: List[str],  # TODO: replace with proper input data type?
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> AsyncGenerator[Optional[AlignmentDatasetSample], None]:
    """Process prompts asynchronously and show tqdm progress bar.

    Output might not be in the same order as input.
    """
    coros = [generate(prompt, model_name, async_semaphore) for prompt in prompts]
    for task in tqdm(asyncio.as_completed(coros), total=len(coros)):
        response = await task
        yield response


# TODO: replace with actual AlignmentDataset.
def write_batch_output(
    output_base_path: str,
    batch_index: int,
    batch_content: List[AlignmentDatasetSample],
    extra_data: Dict[str, Union[str, int]],
) -> None:
    """Write data and metrics to disk."""
    output_file_path = os.path.join(output_base_path, f'output_{batch_index:03d}.json')
    with open(output_file_path, 'w') as output_file:
        output_lines = [json.dumps(item.__dict__) for item in batch_content if item]
        output_file.write('\n'.join(output_lines))

    provenance_file_path = os.path.join(output_base_path, 'provenance.json')

    with open(provenance_file_path, 'w') as provenance_file:
        provenance_data = {'template': TEMPLATE, 'extra_data': extra_data}
        json.dump(provenance_data, provenance_file)
