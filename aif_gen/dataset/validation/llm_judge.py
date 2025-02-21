import asyncio
import logging
from collections import defaultdict
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import backoff
import numpy as np
import openai
import pydantic
from tqdm.asyncio import tqdm

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.typing import Dataset


async def llm_judge_validation(
    dataset: Dataset,
    model_name: str,
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
    dry_run: bool = False,
) -> Optional[List[Optional[Dict[str, float]]]]:
    r"""Use an LLM to judge the quality of the dataset..

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.
        model_name (str): The vLLM-compatible model alias to use for validating the data.
        client (openai.AsyncOpenAI): Handle to openAI client.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        dry_run (bool): If True, validate a dummy sample to ensure the model is setup correctly.

    Returns:
        Optional[List[Optional[Dict[str, float]]]]: For every AlignmentDataset, returns a dictionary with entries of the form '{metric}_{stat}':
            - Stat is one of ['mean', 'median', 'min', 'max']
            - Metric is one of:
                'alignment_chosen'    -> The alignment between the chosen response and prompt, as determined by the LLM.
                'alignment_rejected'  -> The alignment between the rejected response and prompt, as determined by the LLM.
                'coherence_chosen'    -> The coherence in the chosen response, as determined by the LLM.
                'coherence_rejected'  -> The coherence in the rejected response, as determined by the LLM.

    Note:
        - If the dataset is empty, we put None in place of the dictionary.
    """
    if dry_run:
        logging.info(f'Doing dry-run data validation on a single sample...')
        mock_sample = AlignmentDatasetSample('Mock', 'Mock', 'Mock')
        coro = _validate_sample(
            mock_sample,
            client,
            model_name,
            async_semaphore,
            dataset_idx=-1,
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(f'Exception occured on dry-run, skipping validation: {e}')
            raise e
        logging.info('Dry run was a success.')
        return None

    if isinstance(dataset, AlignmentDataset):
        datasets = [dataset]
    else:
        # This assert is here to make mypy happy
        assert isinstance(dataset, ContinualAlignmentDataset)
        datasets = dataset.datasets

    futures = []
    for dataset_idx, dataset in enumerate(datasets):
        dataset_size = len(dataset)
        logging.info(f'Validating Dataset ({dataset_size} samples)')

        for sample in dataset.samples:
            coro = _validate_sample(
                sample,
                client,
                model_name,
                async_semaphore,
                dataset_idx=dataset_idx,
            )
            futures.append(asyncio.create_task(coro))

    try:
        results: List[Dict[str, List[float]]] = [defaultdict(list)] * len(datasets)
        for fut in tqdm.as_completed(futures, total=len(futures)):
            result = await fut
            if result is None:
                continue

            metrics, dataset_idx = result
            for metric_name, metric_value in metrics:
                if metric_value is not None:
                    results[dataset_idx][metric_name].append(metric_value)

        aggregated_results: List[Optional[Dict[str, float]]] = []
        for i, dataset in enumerate(datasets):
            if not len(dataset):
                logging.warning('Skipping LLM judge for empty dataset')
                aggregated_results.append(None)
                continue

            for metric_name, metric_values in results[i].items():
                if len(metric_values) != len(dataset):
                    logging.warning(
                        f'Dataset {i} {metric_name} validation coverage: {len(metric_values)} / {len(dataset)}'
                    )
                if len(metric_values) == 0:
                    raise RuntimeError(
                        f'Could not parse LLM output for any samples in dataset {i}'
                    )
            aggregated_results.append(_compute_statistics(results[i]))
        return aggregated_results

    except BaseException as e:
        logging.exception(f'Exception occured while generating dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None


async def _validate_sample(
    sample: AlignmentDatasetSample,
    client: openai.AsyncOpenAI,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
    dataset_idx: int,
) -> Tuple[Dict[str, Coroutine[Any, Any, Optional[float]]], int]:
    chosen_alignment_prompt = _get_alignment_prompt(sample.prompt, sample.chosen)
    rejected_alignment_prompt = _get_alignment_prompt(sample.prompt, sample.rejected)
    chosen_coherence_prompt = _get_coherence_prompt(sample.chosen)
    rejected_coherence_prompt = _get_coherence_prompt(sample.rejected)

    result = {}
    result['alignment_chosen'] = _get_score(
        chosen_alignment_prompt, client, model_name, async_semaphore
    )
    result['alignment_rejected'] = _get_score(
        rejected_alignment_prompt, client, model_name, async_semaphore
    )
    result['coherence_chosen'] = _get_score(
        chosen_coherence_prompt, client, model_name, async_semaphore
    )
    result['coherence_rejected'] = _get_score(
        rejected_coherence_prompt, client, model_name, async_semaphore
    )
    logging.debug(result)
    return result, dataset_idx


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def _get_score(
    prompt: str,
    client: openai.AsyncOpenAI,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> Optional[float]:
    try:

        class _ValidationResponse(pydantic.BaseModel):
            score: float

        async with async_semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'SyntheticPreference',
                        'schema': _ValidationResponse.model_json_schema(),
                        'strict': True,
                    },
                },
                max_tokens=256,  # TODO: Make this configurable
            )

        model_response = response.choices[0].message.content
        if model_response is None:
            raise ValueError(f'Received None response to prompt: {prompt}')
        assert model_response is not None  # This is for mypy

        score = _ValidationResponse.model_validate_json(model_response).score
        score = max(0, min(1, score))
        logging.debug(f'Prompt: {prompt}, Response: {model_response}, Score: {score}')
        return score

    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None


def _get_alignment_prompt(prompt: str, response: str) -> str:
    return (
        'Please evaluate the alignment of the following response with respect to the given prompt. '
        'Consider whether the response aligns with the prompt in terms of clarity, relevance, and correctness. '
        'Provide a rating from 0 to 1 where 1 indicates perfect alignment and 0 indicates no alignment at all.\n\n'
        f'Prompt: {prompt}\n\n'
        f'Response: {response}\n\n'
        'Alignment Score (0 to 1):'
    )


def _get_coherence_prompt(response: str) -> str:
    return (
        'Please evaluate the coherence of the following response on a scale from 0 to 1, '
        'where 1 indicates excellent coherence and 0 indicates poor coherence:\n\n'
        f'Response: {response}\n\n'
        'Coherence Score (0 to 1):'
    )


def _compute_statistics(results: Dict[str, List[float]]) -> Dict[str, float]:
    statistics: Dict[str, float] = {}
    for metric, values in results.items():
        statistics[f'{metric}_mean'] = float(np.mean(values))
        statistics[f'{metric}_median'] = float(np.median(values))
        statistics[f'{metric}_min'] = float(np.min(values))
        statistics[f'{metric}_max'] = float(np.max(values))
    return statistics
