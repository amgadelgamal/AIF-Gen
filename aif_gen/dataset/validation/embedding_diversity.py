import asyncio
import logging
from collections import defaultdict
from typing import Dict, Generator, Iterable, List, Optional, TypeVar

import backoff
import numpy as np
import openai
from numpy.linalg import norm
from tqdm.asyncio import tqdm

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.generate.caching import AsyncElasticsearchCache
from aif_gen.typing import Dataset


async def llm_embedding_diversity(
    dataset: Dataset,
    model_name: str,
    client: openai.AsyncOpenAI,
    batch_size: int,
    async_semaphore: asyncio.Semaphore,
    dry_run: bool = False,
) -> Optional[List[Optional[Dict[str, float]]]]:
    r"""Use an LLM to judge the quality of the dataset..

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.
        model_name (str): The vLLM-compatible model alias to use for embedding texts.
        client (openai.AsyncOpenAI): Handle to openAI client.
        batch_size (int): Number of items to submit at a time.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        dry_run (bool): If True, validate a dummy sample to ensure the model is setup correctly.

    Returns:
        Optional[List[Optional[Dict[str, float]]]]: For every AlignmentDataset, returns a dictionary with entries of the form '{metric}_{stat}':
            - Stat is one of ['mean', 'median', 'min', 'max']
            - Metric is one of:
                'alignment'           -> Whether the chosen response is more aligned with the prompt compared to the rejected response.
                'coherence_chosen'    -> The coherence in the chosen response, as determined by the LLM.
                'coherence_rejected'  -> The coherence in the rejected response, as determined by the LLM.

    Note:
        - If the dataset is empty, we put None in place of the dictionary.
    """
    cache = await AsyncElasticsearchCache.maybe_from_env_var(
        f'CACHE_VALIDATION_{model_name}'
    )

    if dry_run:
        logging.info(f'Doing dry-run data validation on a single sample...')
        mock_sample = AlignmentDatasetSample('Mock', 'Mock', 'Mock')
        coro = _batch_embed(
            [mock_sample.prompt],
            client=client,
            model_name=model_name,
            async_semaphore=async_semaphore,
            extra_data='prompt',
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(f'Exception occured on dry-run, skipping validation: {e}')
            raise e
        finally:
            if cache is not None:
                await cache.close()

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

        for batch in _batch_iterable(dataset.samples, batch_size=batch_size):
            prompts = [sample.prompt for sample in batch]
            chosen = [sample.chosen for sample in batch]
            rejected = [sample.rejected for sample in batch]

            embed_prompt_coro = _batch_embed(
                prompts,
                client=client,
                model_name=model_name,
                async_semaphore=async_semaphore,
                extra_data='prompt',
            )
            embed_chosen_coro = _batch_embed(
                chosen,
                client=client,
                model_name=model_name,
                async_semaphore=async_semaphore,
                extra_data='chosen',
            )
            embed_rejected_coro = _batch_embed(
                rejected,
                client=client,
                model_name=model_name,
                async_semaphore=async_semaphore,
                extra_data='rejected',
            )
            futures.append(asyncio.create_task(embed_prompt_coro))
            futures.append(asyncio.create_task(embed_chosen_coro))
            futures.append(asyncio.create_task(embed_rejected_coro))

    try:
        results: List[Dict[str, List[List[float]]]] = [defaultdict(list)] * len(
            datasets
        )
        for fut in tqdm.as_completed(futures, total=len(futures)):
            result = await fut
            if result is None:
                continue

            embeddings, text_type = result
            results[dataset_idx][text_type].extend(embeddings)

        aggregated_results: List[Optional[Dict[str, float]]] = []
        for i, dataset in enumerate(datasets):
            if not len(dataset):
                logging.warning('Skipping Embedding Diversity Eval for empty dataset')
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

    finally:
        if cache is not None:
            await cache.close()


T = TypeVar('T')


def _batch_iterable(
    iterable: Iterable[T], batch_size: int
) -> Generator[List[T], None, None]:
    """Splits an iterable into batches of specified size.

    Args:
        iterable: The input iterable to be batched.
        batch_size: The maximum number of items per batch.

    Yields:
        Lists of items, each up to `batch_size` in length.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def _batch_embed(
    texts: list[str],
    client: openai.AsyncOpenAI,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
    extra_data: 'T | None' = None,
) -> tuple[List[List[float]], 'T | None']:
    """Embed text and return extra_data verbatim."""
    async with async_semaphore:
        response = await client.embeddings.create(
            input=texts,
            model=model_name,
            encoding_format='float',
        )
        embeddings: List[List[float]] = [_data.embedding for _data in response.data]
        assert isinstance(embeddings[0][0], float)  # should not be base64-encoded.

    return embeddings, extra_data


def _cosine_similarity_matrix_self_transpose(matrix: np.ndarray) -> np.ndarray:
    """Computes the cosine similarity between each row of a matrix and each column (row of its transpose).

    Args:
        matrix (np.ndarray): A 2D NumPy array of shape (n, m)

    Returns:
        np.ndarray: A 2D array of shape (n, n) where each element (i, j)
                    is the cosine similarity between row i of the matrix and column j of the matrix
                    (i.e., row j of the transpose).
    """
    # Ensure input is a 2D array
    assert matrix.ndim == 2, 'Input must be a 2D array'

    # Compute norms of each row (for the matrix) and each column (same as row of transpose)
    # (n,)
    row_norms = norm(matrix, axis=1)

    # dot product between each row and each column (n, n)
    logging.info(f'Processing matmul: {matrix.shape} @ {matrix.T.shape}')
    dot_product = matrix @ matrix.T

    # Outer product of norms to scale the dot product
    norm_matrix = np.outer(row_norms, row_norms)

    # Avoid division by zero
    norm_matrix[norm_matrix == 0] = 1e-10

    # Cosine similarity matrix
    cosine_sim = dot_product / norm_matrix

    return cosine_sim


def _compute_statistics(results: Dict[str, List[List[float]]]) -> Dict[str, float]:
    statistics: Dict[str, float] = {}
    for metric, values in results.items():
        embeddings = np.asarray(values)  # (dataset, embed_dim)

        # (dataset, dataset)
        similarity_pairwise = _cosine_similarity_matrix_self_transpose(embeddings)
        similarity = similarity_pairwise.mean(axis=-1)  # (dataset,)

        statistics[f'{metric}_mean'] = float(np.mean(similarity))
        statistics[f'{metric}_median'] = float(np.median(similarity))
        statistics[f'{metric}_min'] = float(np.min(similarity))
        statistics[f'{metric}_max'] = float(np.max(similarity))
    return statistics
