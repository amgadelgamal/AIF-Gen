import asyncio
import logging
from typing import Any, Dict, Optional

import backoff
import openai
import pydantic
from tqdm.asyncio import tqdm

from aif_gen.api.prompt_mapper import PromptMapper
from aif_gen.api.response_mapper import ResponseMapper
from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.task.alignment_task import AlignmentTask


async def generate_continual_dataset(
    config_dict: Dict[str, Any],  # TODO: Should bind this type
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
) -> Optional[ContinualAlignmentDataset]:
    r"""Generate a ContinualAlignmentDataset dataset given the AlignmentTask, and model.

    Args:
        config_dict (Dict[str, Any]): Configuration file storing tasks specifications and model info.
        client (openai.AsyncOpenAI): Handle to openAI client.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.

    Returns:
        Optional[ContinualAlignmentDataset]: The synthetically generated dataset.
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
    futures = [asyncio.create_task(coro) for coro in coros]

    try:
        datasets = []
        for fut in tqdm.as_completed(futures, total=len(futures)):
            dataset = await fut
            if dataset is not None:
                datasets.append(dataset)
        return ContinualAlignmentDataset(datasets)
    except BaseException as e:
        logging.exception(f'Exception occured while generating continual dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None


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

    Raises:
        pydantic.ValidationError: If the response cannot be parsed according to the structured json schema.
        openai.NotFoundError: If the openAI model cannot be accessed at the configured endpoint.
    """
    logging.info(f'Generating AIF Dataset for task: {task}')

    prompt_mapper = PromptMapper()
    response_mapper = ResponseMapper()

    coros = [
        generate_sample(
            task, client, model_name, prompt_mapper, response_mapper, async_semaphore
        )
        for _ in range(num_samples)
    ]
    futures = [asyncio.create_task(coro) for coro in coros]

    try:
        samples = []
        for fut in tqdm.as_completed(futures, total=len(futures)):
            sample = await fut
            if sample is not None:
                samples.append(sample)
        if len(samples) != num_samples:
            logging.warning(
                f'Requested {num_samples} response pairs LM generated {len(samples)}'
            )
        return AlignmentDataset(task, samples)
    except openai.NotFoundError as e:
        logging.error(f'Could not connect to openAI model: error: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None
    except BaseException as e:
        logging.exception(f'Exception occured while generating dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def generate_sample(
    task: AlignmentTask,
    client: openai.AsyncOpenAI,
    model_name: str,
    prompt_mapper: PromptMapper,
    response_mapper: ResponseMapper,
    async_semaphore: asyncio.Semaphore,
) -> Optional[AlignmentDatasetSample]:
    r"""Generate a AlignmentDataset dataset given the AlignmentTask, and model.

    Args:
        task (AlignmentTask): The AlignmentTask to generate data for.
        client (openai.AsyncOpenAI): Handle to openAI client.
        model_name (str): openAI model alias.
        prompt_mapper (PromptMapper): Creates the 'meta-prompt' for this sample's task prompt.
        response_mapper (ResponseMapper): Created the 'meta-prompt' for this sample's response prompt.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.

    Returns:
        Optional[AlignmentDatasetSample]: A single sample of the dataset (None if pydantic.ValidationError occurred).

    Raises:
        openai.NotFoundError: If the openAI model cannot be accessed at the configured endpoint.
    """
    try:

        class _ResponsePair(pydantic.BaseModel):
            chosen: str
            rejected: str

        meta_prompt = prompt_mapper.generate_prompt(task)

        async with async_semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': meta_prompt}],
            )

        prompt = response.choices[0].message.content
        if prompt is None:
            raise ValueError(f'Received None response to prompt: {meta_prompt}')
        assert prompt is not None  # This is for mypy

        task_prompt = response_mapper.generate_prompt(task, prompt)

        async with async_semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': task_prompt}],
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
        sample = AlignmentDatasetSample(
            prompt,
            chosen=structured_response.chosen,
            rejected=structured_response.rejected,
        )
        logging.debug(f'Meta Prompt: {meta_prompt}, Sample: {sample}')
        return sample
    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None
