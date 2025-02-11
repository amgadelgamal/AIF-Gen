import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

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

    prompt_mapper = PromptMapper()
    response_mapper = ResponseMapper()

    task_specs = config_dict['data']['task_specs']
    futures, tasks, dataset_sizes = [], [], []
    for dataset_idx, task_spec in enumerate(task_specs):
        task = AlignmentTask.from_dict(task_spec['alignment_task'])
        dataset_size = task_spec['num_samples']
        logging.info(f'Generating Dataset ({dataset_size} samples) for task: {task}')

        tasks.append(task)
        dataset_sizes.append(dataset_size)
        for _ in range(dataset_size):
            coro = generate_sample(
                task,
                client,
                model_name,
                prompt_mapper,
                response_mapper,
                async_semaphore,
                dataset_idx=dataset_idx,
            )
            futures.append(asyncio.create_task(coro))

    try:
        samples: List[List[AlignmentDatasetSample]] = [[] for _ in range(len(futures))]
        for fut in tqdm.as_completed(futures, total=len(futures)):
            result = await fut
            if result is not None:
                dataset_idx, sample = result
                samples[dataset_idx].append(sample)

        continual_dataset = ContinualAlignmentDataset(datasets=[])
        for i in range(len(futures)):
            if len(samples[i]) != dataset_sizes[i]:
                logging.warning(
                    f'Dataset {i} requested {dataset_sizes[i]} response pairs but LM generated {len(samples[i])}'
                )
            continual_dataset.append(AlignmentDataset(tasks[i], samples[i]))
        return continual_dataset
    except BaseException as e:
        logging.exception(f'Exception occured while generating continual dataset: {e}')
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
    dataset_idx: int,
) -> Optional[Tuple[AlignmentDatasetSample, int]]:
    r"""Generate a AlignmentDataset dataset given the AlignmentTask, and model.

    Args:
        task (AlignmentTask): The AlignmentTask to generate data for.
        client (openai.AsyncOpenAI): Handle to openAI client.
        model_name (str): openAI model alias.
        prompt_mapper (PromptMapper): Creates the 'meta-prompt' for this sample's task prompt.
        response_mapper (ResponseMapper): Created the 'meta-prompt' for this sample's response prompt.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        dataset_idx (int): The idx of the dataset that the sample is requested for to align out-of-order asyn execution.

    Returns:
        Optional[Tuple[AlignmentDatasetSample, int]]: A single sample of the dataset, and the dataset idx (None if pydantic.ValidationError occurred).

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
        return sample, dataset_idx
    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None
