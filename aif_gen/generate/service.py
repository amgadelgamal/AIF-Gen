import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List

import backoff
import openai
import pydantic
from tqdm.asyncio import tqdm

from aif_gen.api.prompt_mapper import PromptMapper
from aif_gen.api.response_mapper import ResponseMapper
from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample
from aif_gen.task.alignment_task import AlignmentTask

# TODO: Wrap the openAI API and integrate VLLM API
try:
    client = openai.AsyncOpenAI()
except (openai.OpenAIError, Exception) as e:
    logging.exception(e)


async def process_tasks(
    config_dict: Dict[str, Any],  # TODO: Should bind this type
    async_semaphore: asyncio.Semaphore,
) -> AsyncGenerator[AlignmentDataset, None]:
    r"""Generate a ContinualAlignment dataset given the AIF configuratiohn file.

    Args:
        config_dict (Dict[str, Any]): Configuration file storing tasks specifications and model info.
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
            model_name,
            async_semaphore,
        )
        for task_spec in task_specs
    ]
    for coro in tqdm(asyncio.as_completed(coros), total=len(coros)):
        dataset = await coro
        yield dataset


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def generate_dataset(
    task: AlignmentTask,
    num_samples: int,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> AlignmentDataset:
    r"""Generate a AlignmentDataset dataset given the AlignmentTask, and model.

    Args:
        task (AlignmentTask): The AlignmentTask to generate data for.
        num_samples (int): The number of samples to generate in the dataset.
        model_name (str): openAI model alias.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.

    Returns:
        AlignmentDataset: The synthetically generated AlignmentDataset.
    """
    logging.info(f'Generating AIF Dataset for task: {task}')
    prompt = await _generate_task_prompt(task, num_samples, model_name, async_semaphore)

    class _ResponsePairList(pydantic.BaseModel):
        class _ResponsePair(pydantic.BaseModel):
            chosen: str
            rejected: str

        responses: List[_ResponsePair]

    async with async_semaphore:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'SyntheticPreference',
                    'schema': _ResponsePairList.model_json_schema(),
                    'strict': True,
                },
            },
        )
    output = response.choices[0].message.content

    samples = []
    try:
        responses = _ResponsePairList.model_validate_json(output).responses
        logging.debug(f'Received {len(responses)} response pairs: {responses}')
        if len(responses) != num_samples:
            logging.warning(
                f'Requested {num_samples} response pairs LM generated {len(responses)}'
            )

        samples = [
            AlignmentDatasetSample(
                prompt=prompt, chosen=response.chosen, rejected=response.rejected
            )
            for response in responses
        ]
    except pydantic.ValidationError as e:
        logging.exception(e)

    # TODO: May want to include data splits here directly #44
    return AlignmentDataset(task, samples)


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def _generate_task_prompt(
    task: AlignmentTask,
    num_samples: int,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> str:
    logging.info(f'Generating task prompt for {task}')
    prompt_mapper = PromptMapper()
    meta_prompt = prompt_mapper.generate_prompt(task)
    logging.debug(f'Using meta prompt: {meta_prompt}')

    async with async_semaphore:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': meta_prompt}],
        )
    prompt = response.choices[0].message.content
    assert prompt is not None
    logging.info(f'Generated prompt: {prompt}')

    logging.info(f'Generating {num_samples} response pairs for {task}')
    response_mapper = ResponseMapper()
    task_prompt = response_mapper.generate_prompt(task, prompt, num_samples)
    logging.debug(f'Using task prompt: {task_prompt}')
    return task_prompt
