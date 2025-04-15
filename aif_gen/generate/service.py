import asyncio
import logging
import os
from functools import lru_cache
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
from aif_gen.generate.caching import AsyncElasticsearchCache
from aif_gen.task.alignment_task import AlignmentTask


async def generate_continual_dataset(
    data_config: Dict[str, Any],  # TODO: Should bind this type
    model_name: str,
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
    max_tokens_prompt_response: int = 1024,
    max_tokens_chosen_rejected_response: int = 2048,
    dry_run: bool = False,
) -> Optional[ContinualAlignmentDataset]:
    r"""Generate a ContinualAlignmentDataset dataset given the AlignmentTask, and model.

    Args:
        data_config (Dict[str, Any]): Configuration file storing tasks specifications and model info.
        model_name (str): The vLLM-compatible model alias to use for generation synthetic samples.
        client (openai.AsyncOpenAI): Handle to openAI client.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens_prompt_response (int): Configurable limit on the max_tokens for the generated prompt response.
        max_tokens_chosen_rejected_response (int): Configurable limit on the max_tokens for the generated chosen and rejected response.
        dry_run (bool): If True, ignore the config and generate a dummy sample to ensure the model is setup correctly.

    Returns:
        Optional[ContinualAlignmentDataset]: The synthetically generated dataset.
    """
    prompt_mapper = PromptMapper()
    response_mapper = ResponseMapper()

    task_specs = data_config['task_specs']

    if dry_run:
        logging.info(f'Doing dry-run data generation on a single sample...')
        mock_task = AlignmentTask.from_dict(task_specs[0]['alignment_task'])
        cache = await AsyncElasticsearchCache.maybe_from_env_var(
            index_name=f'CACHE_DATA_GENERATION_{model_name}'
        )
        coro = _generate_sample(
            mock_task,
            client,
            model_name,
            prompt_mapper,
            response_mapper,
            async_semaphore,
            max_tokens_prompt_response,
            max_tokens_chosen_rejected_response,
            dataset_idx=-1,
            prompt_idx=-1,
            cache=cache,
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(f'Exception occured on dry-run, skipping generation: {e}')
            raise e
        finally:
            if cache is not None:
                await cache.close()

        logging.info('Dry run was a success.')
        return None

    cache = await AsyncElasticsearchCache.maybe_from_env_var(
        index_name=f'CACHE_DATA_GENERATION_{model_name}'
    )
    futures, tasks, dataset_sizes = [], [], []
    for dataset_idx, task_spec in enumerate(task_specs):
        task = AlignmentTask.from_dict(task_spec['alignment_task'])
        dataset_size = task_spec['num_samples']
        logging.info(f'Generating Dataset ({dataset_size} samples) {task}')

        tasks.append(task)
        dataset_sizes.append(dataset_size)
        for _sample_idx in range(dataset_size):
            coro = _generate_sample(
                task,
                client,
                model_name,
                prompt_mapper,
                response_mapper,
                async_semaphore,
                max_tokens_prompt_response,
                max_tokens_chosen_rejected_response,
                dataset_idx=dataset_idx,
                prompt_idx=_sample_idx,
                cache=cache,
            )
            futures.append(asyncio.create_task(coro))

    try:
        samples: List[List[AlignmentDatasetSample]] = [
            [] for _ in range(len(dataset_sizes))
        ]
        for fut in tqdm.as_completed(futures, total=len(futures)):
            result = await fut
            if result is not None:
                sample, dataset_idx = result
                samples[dataset_idx].append(sample)

        continual_dataset = ContinualAlignmentDataset(datasets=[])
        for i in range(len(samples)):
            if len(samples[i]) != dataset_sizes[i]:
                logging.warning(
                    f'Dataset {i} requested {dataset_sizes[i]} samples but LM generated {len(samples[i])}'
                )
            continual_dataset.append(AlignmentDataset(tasks[i], samples[i]))
        return continual_dataset
    except BaseException as e:
        logging.exception(f'Exception occured while generating dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None
    finally:
        if cache is not None:
            await cache.close()


@lru_cache(maxsize=None)
def get_tries(default: int = 3) -> int:
    if 'BACKOFF_RETRIES' in os.environ:
        try:
            return int(os.environ['BACKOFF_RETRIES'])
        except:
            logging.warning(f'Failed to parse BACKOFF_RETRIES, using: {default}')
    return default


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError),
    max_tries=get_tries(),
)
async def _generate_sample(
    task: AlignmentTask,
    client: openai.AsyncOpenAI,
    model_name: str,
    prompt_mapper: PromptMapper,
    response_mapper: ResponseMapper,
    async_semaphore: asyncio.Semaphore,
    max_tokens_prompt_response: int,
    max_tokens_chosen_rejected_response: int,
    dataset_idx: int,
    prompt_idx: int,
    cache: 'AsyncElasticsearchCache | None' = None,
) -> Optional[Tuple[AlignmentDatasetSample, int]]:
    r"""Generate a AlignmentDataset dataset given the AlignmentTask, and model.

    Args:
        task (AlignmentTask): The AlignmentTask to generate data for.
        client (openai.AsyncOpenAI): Handle to openAI client.
        model_name (str): openAI model alias.
        prompt_mapper (PromptMapper): Creates the 'meta-prompt' for this sample's task prompt.
        response_mapper (ResponseMapper): Created the 'meta-prompt' for this sample's response prompt.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens_prompt_response (int): Configurable limit on the max_tokens for the generated prompt response.
        max_tokens_chosen_rejected_response (int): Configurable limit on the max_tokens for the generated chosen and rejected response.
        dataset_idx (int): The idx of the dataset that the sample is requested for to align out-of-order asyn execution.
        max_tokens (int): Max number of tokens to generate.
        prompt_idx (int): The idx of the sample, to distinguish between multiple requests for the same task.
        cache (AsyncElasticsearchCache): Optionally specify a AsyncElasticsearchCache instance for caching.

    Returns:
        Optional[Tuple[AlignmentDatasetSample, int]]: A single sample of the dataset, and the dataset idx (None if pydantic.ValidationError occurred).

    Raises:
        openai.NotFoundError: If the openAI model cannot be accessed at the configured endpoint.
    """
    try:

        class _PromptProposal(pydantic.BaseModel, extra='forbid'):
            prompt: str

        class _ResponsePair(pydantic.BaseModel, extra='forbid'):
            chosen: str
            rejected: str

        meta_prompt = prompt_mapper.generate_prompt(task)
        meta_prompt_nonce = f'{prompt_idx}'

        async with async_semaphore:
            if cache is not None:
                output = await cache.get(meta_prompt, nonce=meta_prompt_nonce)
            else:
                output = None

            if output is None:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': meta_prompt}],
                    max_tokens=max_tokens_prompt_response,
                    response_format={
                        'type': 'json_schema',
                        'json_schema': {
                            'name': 'PromptProposal',
                            'schema': _PromptProposal.model_json_schema(),
                            'strict': True,
                        },
                    },
                )
                output = response.choices[0].message.content
                assert output is not None  # This is for mypy

        if output is None:
            raise ValueError(f'Received None response to prompt: {meta_prompt}')

        prompt = _PromptProposal.model_validate_json(output).prompt

        # Update/set cache only after validating output JSON.
        if cache is not None:
            await cache.set(query=meta_prompt, value=output, nonce=meta_prompt_nonce)

        task_prompt = response_mapper.generate_prompt(task, prompt)
        logging.debug(
            f'Meta Prompt: {meta_prompt} (Nonce: {meta_prompt_nonce}), '
            f'Model Response: {prompt}'
        )

        async with async_semaphore:
            if cache is not None:
                output = await cache.get(task_prompt)
            else:
                output = None

            if output is None:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': task_prompt}],
                    max_tokens=max_tokens_chosen_rejected_response,
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
                assert output is not None  # This is for mypy

        if prompt is None:
            raise ValueError(f'Received None response to prompt: {prompt}')

        structured_response = _ResponsePair.model_validate_json(output)
        if cache is not None:
            await cache.set(query=task_prompt, value=output)

        sample = AlignmentDatasetSample(
            prompt,
            chosen=structured_response.chosen,
            rejected=structured_response.rejected,
        )
        logging.debug(f'Task Prompt: {task_prompt}, Sample: {sample}')
        return sample, dataset_idx
    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None


async def transmute_continual_dataset(
    input_dataset: ContinualAlignmentDataset,
    model_name: str,
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
    max_tokens_chosen_rejected_response: int = 2048,
    dry_run: bool = False,
) -> Optional[ContinualAlignmentDataset]:
    r"""Overriding the 'rejected' responses with the 'chosen' response of a new model.

    Args:
        input_dataset (ContinualAlignmentDataset): Dataset to transmute.
        model_name (str): The vLLM-compatible model alias to use for generation synthetic samples.
        client (openai.AsyncOpenAI): Handle to openAI client.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens_chosen_rejected_response (int): Configurable limit on the max_tokens for the generated chosen and rejected response.
        dry_run (bool): If True, ignore the config and generate a dummy sample to ensure the model is setup correctly.

    Returns:
        Optional[ContinualAlignmentDataset]: The synthetically generated dataset.
    """
    response_mapper = ResponseMapper()

    if dry_run:
        logging.info(f'Doing dry-run data transmutation on a single sample...')
        mock_task = input_dataset.datasets[0].task
        mock_sample = input_dataset[0]
        cache = await AsyncElasticsearchCache.maybe_from_env_var(
            index_name=f'CACHE_DATA_TRANSMUTATION_{model_name}'
        )
        coro = _transmute_sample(
            mock_task,
            mock_sample,  # type: ignore
            client,
            model_name,
            response_mapper,
            async_semaphore,
            max_tokens_chosen_rejected_response,
            dataset_idx=-1,
            cache=cache,
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(f'Exception occured on dry-run, skipping transmute: {e}')
            raise e
        finally:
            if cache is not None:
                await cache.close()

        logging.info('Dry run was a success.')
        return None

    cache = await AsyncElasticsearchCache.maybe_from_env_var(
        index_name=f'CACHE_DATA_TRANSMUTATION_{model_name}'
    )
    futures, tasks, dataset_sizes = [], [], []
    for dataset_idx, dataset in enumerate(input_dataset.datasets):
        task = dataset.task
        dataset_size = len(dataset)
        logging.info(f'Transmuting Dataset ({dataset_size} samples) {task}')

        tasks.append(task)
        dataset_sizes.append(dataset_size)
        for _sample_idx in range(dataset_size):
            coro = _transmute_sample(
                task,
                dataset[_sample_idx],  # type: ignore
                client,
                model_name,
                response_mapper,
                async_semaphore,
                max_tokens_chosen_rejected_response,
                dataset_idx=dataset_idx,
                cache=cache,
            )
            futures.append(asyncio.create_task(coro))

    try:
        samples: List[List[AlignmentDatasetSample]] = [
            [] for _ in range(len(dataset_sizes))
        ]
        for fut in tqdm.as_completed(futures, total=len(futures)):
            result = await fut
            if result is not None:
                sample, dataset_idx = result
                samples[dataset_idx].append(sample)

        continual_dataset = ContinualAlignmentDataset(datasets=[])
        for i in range(len(samples)):
            if len(samples[i]) != dataset_sizes[i]:
                logging.warning(
                    f'Dataset {i} requested {dataset_sizes[i]} samples but LM generated {len(samples[i])}'
                )
            continual_dataset.append(AlignmentDataset(tasks[i], samples[i]))
        return continual_dataset
    except BaseException as e:
        logging.exception(f'Exception occured while generating dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None
    finally:
        if cache is not None:
            await cache.close()


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError),
    max_tries=get_tries(),
)
async def _transmute_sample(
    task: AlignmentTask,
    sample: AlignmentDatasetSample,
    client: openai.AsyncOpenAI,
    model_name: str,
    response_mapper: ResponseMapper,
    async_semaphore: asyncio.Semaphore,
    max_tokens_chosen_rejected_response: int,
    dataset_idx: int,
    cache: 'AsyncElasticsearchCache | None' = None,
) -> Optional[Tuple[AlignmentDatasetSample, int]]:
    r"""Transmute an AlignmentDatasetSample given the model.

    Args:
        task (AlignmentTask): The AlignmentTask to transmute.
        sample (AlignmentDatasetSample): The AlignmentDatasetSample to transmute.
        client (openai.AsyncOpenAI): Handle to openAI client.
        model_name (str): openAI model alias.
        response_mapper (ResponseMapper): Created the 'meta-prompt' for this sample's response prompt.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens_chosen_rejected_response (int): Configurable limit on the max_tokens for the generated chosen and rejected response.
        dataset_idx (int): The idx of the dataset that the sample is requested for to align out-of-order asyn execution.
        max_tokens (int): Max number of tokens to generate.
        cache (AsyncElasticsearchCache): Optionally specify a AsyncElasticsearchCache instance for caching.

    Returns:
        Optional[Tuple[AlignmentDatasetSample, int]]: A single sample of the dataset, and the dataset idx (None if pydantic.ValidationError occurred).

    Raises:
        openai.NotFoundError: If the openAI model cannot be accessed at the configured endpoint.
    """
    try:

        class _ResponsePair(pydantic.BaseModel, extra='forbid'):
            chosen: str
            rejected: str

        prompt = sample.prompt
        task_prompt = response_mapper.generate_prompt(task, prompt)

        async with async_semaphore:
            if cache is not None:
                output = await cache.get(task_prompt)
            else:
                output = None

            if output is None:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': task_prompt}],
                    max_tokens=max_tokens_chosen_rejected_response,
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
                assert output is not None  # This is for mypy

        if prompt is None:
            raise ValueError(f'Received None response to prompt: {prompt}')

        structured_response = _ResponsePair.model_validate_json(output)
        if cache is not None:
            await cache.set(query=task_prompt, value=output)

        # The models 'chosen' response becomes the samples 'rejected' attribute
        sample.rejected = structured_response.chosen
        logging.debug(f'Task Prompt: {task_prompt}, Sample: {sample}')
        return sample, dataset_idx
    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None


async def filter_continual_alignment_dataset_style_normalize(
    input_dataset: ContinualAlignmentDataset,
    model_name: str,
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
    max_tokens: int = 4096,
    dry_run: bool = False,
) -> Optional[ContinualAlignmentDataset]:
    r"""Normalize the style between chosen/rejected responses while preserving quality differences.

    Args:
        input_dataset (ContinualAlignmentDataset): Dataset to filter and normalize.
        model_name (str): The model alias to use for generating normalized responses.
        client (openai.AsyncOpenAI): Handle to openAI client.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens (int): Configurable limit on the max_tokens for the generated responses.
        dry_run (bool): If True, process a single sample to ensure the model is setup correctly.

    Returns:
        Optional[ContinualAlignmentDataset]: The filtered and normalized dataset.
    """
    if dry_run:
        logging.info(f'Doing dry-run style normalization on a single sample...')
        mock_task = input_dataset.datasets[0].task
        mock_sample = input_dataset[0]
        cache = await AsyncElasticsearchCache.maybe_from_env_var(
            index_name=f'CACHE_STYLE_NORMALIZATION_{model_name}'
        )
        coro = _normalize_sample_style(
            mock_task,
            mock_sample,  # type: ignore
            client,
            model_name,
            async_semaphore,
            max_tokens,
            dataset_idx=-1,
            cache=cache,
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(
                f'Exception occurred on dry-run, skipping normalization: {e}'
            )
            raise e
        finally:
            if cache is not None:
                await cache.close()

        logging.info('Dry run was a success.')
        return None

    cache = await AsyncElasticsearchCache.maybe_from_env_var(
        index_name=f'CACHE_STYLE_NORMALIZATION_{model_name}'
    )
    futures, tasks, dataset_sizes = [], [], []
    for dataset_idx, dataset in enumerate(input_dataset.datasets):
        task = dataset.task
        dataset_size = len(dataset)
        logging.info(f'Style normalizing Dataset ({dataset_size} samples) {task}')

        tasks.append(task)
        dataset_sizes.append(dataset_size)
        for _sample_idx in range(dataset_size):
            coro = _normalize_sample_style(
                task,
                dataset[_sample_idx],  # type: ignore
                client,
                model_name,
                async_semaphore,
                max_tokens,
                dataset_idx=dataset_idx,
                cache=cache,
            )
            futures.append(asyncio.create_task(coro))

    try:
        samples: List[List[AlignmentDatasetSample]] = [
            [] for _ in range(len(dataset_sizes))
        ]
        for fut in tqdm.as_completed(futures, total=len(futures)):
            result = await fut
            if result is not None:
                sample, dataset_idx = result
                samples[dataset_idx].append(sample)

        continual_dataset = ContinualAlignmentDataset(datasets=[])
        for i in range(len(samples)):
            if len(samples[i]) != dataset_sizes[i]:
                logging.warning(
                    f'Dataset {i} requested {dataset_sizes[i]} samples but LM generated {len(samples[i])}'
                )
            continual_dataset.append(AlignmentDataset(tasks[i], samples[i]))
        return continual_dataset
    except BaseException as e:
        logging.exception(f'Exception occurred while normalizing dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None
    finally:
        if cache is not None:
            await cache.close()


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError),
    max_tries=get_tries(),
)
async def _normalize_sample_style(
    task: AlignmentTask,
    sample: AlignmentDatasetSample,
    client: openai.AsyncOpenAI,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
    max_tokens: int,
    dataset_idx: int,
    cache: 'AsyncElasticsearchCache | None' = None,
) -> Optional[Tuple[AlignmentDatasetSample, int]]:
    r"""Normalize the style between chosen/rejected responses in a single sample.

    Args:
        task (AlignmentTask): The AlignmentTask for context.
        sample (AlignmentDatasetSample): The sample to normalize.
        client (openai.AsyncOpenAI): Handle to openAI client.
        model_name (str): openAI model alias.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens (int): Max number of tokens to generate.
        dataset_idx (int): The idx of the dataset that the sample is from.
        cache (AsyncElasticsearchCache): Optionally specify a cache instance for caching.

    Returns:
        Optional[Tuple[AlignmentDatasetSample, int]]: The normalized sample and dataset index.

    Raises:
        openai.NotFoundError: If the openAI model cannot be accessed at the configured endpoint.
    """
    try:

        class _ResponsePair(pydantic.BaseModel, extra='forbid'):
            chosen: str
            rejected: str

        prompt = (
            'Given the following RLHF data point (chosen, reject), rewrite both so that they do not differ in style '
            '(e.g., punctuation, obvious wordings), but preserve the difference in quality and the key idea in each. '
            "The 'chosen' should be better, but both should be similar in style. "
            "Return the 'chosen' and 'rejected' fields.\n\n"
            f'Response 1 (chosen):\n{sample.chosen}\n\nResponse 2 (rejected):\n{sample.rejected}'
        )

        cache_key = f'style_norm_{str(task.domain) + task.objective + task.preference}_{hash(sample.chosen + sample.rejected)}'

        async with async_semaphore:
            if cache is not None:
                output = await cache.get(cache_key)
            else:
                output = None

            if output is None:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
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
                assert output is not None  # This is for mypy

        if output is None:
            raise ValueError(f'Received None response to prompt: {prompt}')

        structured_response = _ResponsePair.model_validate_json(output)
        if cache is not None:
            await cache.set(query=cache_key, value=output)

        # Update the sample with normalized responses
        sample.chosen = structured_response.chosen
        sample.rejected = structured_response.rejected

        logging.debug(f'Normalized sample: {sample}')
        return sample, dataset_idx
    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None
