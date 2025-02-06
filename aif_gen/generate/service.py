import asyncio
import logging
from typing import AsyncGenerator, List, Optional

import backoff
import openai
import pydantic
from tqdm.asyncio import tqdm

from aif_gen.dataset import AlignmentDatasetSample


class _OutputModel(pydantic.BaseModel):
    """Synthetic preference based on the given principles."""

    chosen: str
    rejected: str


try:
    client = openai.AsyncOpenAI()
except Exception as e:
    logging.exception(e)


async def process_prompts(
    prompts: List[str],
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> AsyncGenerator[Optional[AlignmentDatasetSample], None]:
    """Process prompts asynchronously. Output might not be in the same order as input."""
    coros = [generate(prompt, model_name, async_semaphore) for prompt in prompts]
    for task in tqdm(asyncio.as_completed(coros), total=len(coros)):
        response = await task
        yield response


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def generate(
    prompt: str,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> Optional[AlignmentDatasetSample]:
    """ChatCompletion generation. Returns None if output is not parsed."""
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
        logging.debug(f'Received response: {response}')
        return AlignmentDatasetSample(
            prompt=prompt, chosen=response.chosen, rejected=response.rejected
        )
    except pydantic.ValidationError as e:
        logging.exception(e)
        return None
