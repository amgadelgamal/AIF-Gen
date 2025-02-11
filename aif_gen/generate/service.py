import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional

import backoff
import openai
import pydantic
from tqdm.asyncio import tqdm

from aif_gen.api.prompt_mapper import PromptMapper
from aif_gen.api.response_mapper import ResponseMapper
from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample
from aif_gen.task.alignment_task import AlignmentTask


async def process_tasks(
    config_dict: Dict[str, Any],  # TODO: Should bind this type
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
) -> AsyncGenerator[AlignmentDataset, None]:
    r"""Generate a ContinualAlignment dataset given the AIF configuratiohn file.

    Args:
        config_dict (Dict[str, Any]): Configuration file storing tasks specifications and model info.
        client (openai.AsyncOpenAI): Handle to openAI client.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.

    Returns:
        AsyncGenerator[AlignmentDataset, None]: Yields each slice of the ContinualAlignmentDataset.
    """
    model_name = config_dict['model_name']
    logging.info(f'Using Model: {model_name}')

    task_specs = config_dict['data']['task_specs']
    coros = [
        generate_dataset(
            AlignmentTask.from_dict(task_spec['alignment_task']),
            task_spec['num_samples'],
            client,
            model_name,
            async_semaphore,
        )
        for task_spec in task_specs
    ]
    for coro in tqdm(asyncio.as_completed(coros), total=len(coros)):
        dataset = await coro
        if dataset is not None:
            yield dataset


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def generate_dataset(
    task: AlignmentTask,
    num_samples: int,
    client: openai.AsyncOpenAI,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> Optional[AlignmentDataset]:
    r"""Generate a AlignmentDataset dataset given the AlignmentTask, and model.

    Args:
        task (AlignmentTask): The AlignmentTask to generate data for.
        num_samples (int): The number of samples to generate in the dataset.
        client (openai.AsyncOpenAI): Handle to openAI client.
        model_name (str): openAI model alias.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.

    Returns:
        Optional[AlignmentDataset]: The synthetically generated AlignmentDataset.
    """
    logging.info(f'Generating AIF Dataset for task: {task}')

    prompt_mapper = PromptMapper()
    response_mapper = ResponseMapper()

    class _ResponsePair(pydantic.BaseModel):
        chosen: str
        rejected: str

    samples = []
    for _ in range(num_samples):
        meta_prompt = prompt_mapper.generate_prompt(task)
        logging.debug(f'Using meta prompt: {meta_prompt}')

        try:
            async with async_semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': meta_prompt}],
                )

            prompt = response.choices[0].message.content
            if prompt is None:
                raise ValueError(f'Received None response to prompt: {meta_prompt}')
            assert prompt is not None  # This is for mypy
            logging.debug(f'Generated prompt: {prompt}')

            task_prompt = response_mapper.generate_prompt(task, prompt)
            logging.debug(f'Using task prompt: {task_prompt}')

            async with async_semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    response_format={
                        'type': 'json_schema',
                        'json_schema': {
                            'name': 'SyntheticPreference',
                            'schema': _ResponsePair.model_json_schema(),
                            'strict': True,
                        },
                    },
                )

            output = response.choices[0].message.content
            if prompt is None:
                raise ValueError(f'Received None response to prompt: {prompt}')
            assert output is not None  # This is for mypy
            structured_response = _ResponsePair.model_validate_json(output)

        except pydantic.ValidationError as e:
            logging.error(f'Failed to bind structured output json schema: {e}')
            continue
        except openai.NotFoundError as e:
            logging.error(f'Could not connect to openAI model: error: {e}')
            return None
        except Exception as e:
            logging.exception(f'Exception occured while generating sample: {e}')
            return None

        logging.debug(f'Received response pairs: {structured_response}')
        sample = AlignmentDatasetSample(
            prompt,
            chosen=structured_response.chosen,
            rejected=structured_response.rejected,
        )
        samples.append(sample)

    if len(samples) != num_samples:
        logging.warning(
            f'Requested {num_samples} response pairs LM generated {len(samples)}'
        )

    return AlignmentDataset(task, samples)
