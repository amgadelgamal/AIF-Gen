import logging
from typing import Callable, Dict, List, Optional

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset


def diversity_validation(
    dataset: Dataset, ngram: int = 3
) -> List[Optional[Dict[str, float]]]:
    r"""Report the inverse Self-BLEU score as a measure of diversity within the generated samples.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.
        ngram (int): The maximum n-gram order for BLEU calculation. Default of 3 matches the original paper.

    Returns:
        List[Optional[Dict[str, float]]]: For every AlignmentDataset, returns a dictionary with entries of the form '{metric}_stat':
            - Stat is one of ['mean', 'median', 'min', 'max']
            - Metric is one of:
                'prompt_diversity'    -> The diversity across prompts in samples of the AlignmentDataset.
                'chosen_diversity'    -> The diversity across chosen responses in samples of the AlignmentDataset.
                'rejected_diversity'  -> The diversity across rejected responses in samples of the AlignmentDataset.

    Note:
        - If the dataset is empty, we put None in place of the dictionary.

    References:
        - https://arxiv.org/pdf/1802.01886
    """
    if not (isinstance(ngram, int) and ngram > 0):
        raise ValueError(f'ngram must be a positive integer, got: {ngram}')

    _download_nltk_resources()

    if isinstance(dataset, AlignmentDataset):
        datasets = [dataset]
    else:
        # This assert is here to make mypy happy
        assert isinstance(dataset, ContinualAlignmentDataset)
        datasets = dataset.datasets

    results: List[Optional[Dict[str, float]]] = []
    for dataset in datasets:
        if len(dataset):
            result = _diversity_validation(dataset, ngram)
        else:
            logging.warning(f'Skipping diversity on empty dataset: {dataset}')
            result = None
        results.append(result)
    return results


def _diversity_validation(dataset: AlignmentDataset, ngram: int) -> Dict[str, float]:
    weight = [1.0 / ngram for _ in range(ngram)]
    prompts = [sample.prompt for sample in dataset.samples]
    chosens = [sample.chosen for sample in dataset.samples]
    rejected = [sample.rejected for sample in dataset.samples]

    results: Dict[str, List[float]] = {}
    results['prompt_diversity'] = _compute_diversity(prompts, weight)
    results['chosen_diversity'] = _compute_diversity(chosens, weight)
    results['rejected_diversity'] = _compute_diversity(rejected, weight)
    return _compute_statistics(results)


def _compute_diversity(response_set: List[str], weight: List[float]) -> List[float]:
    if 0 <= len(response_set) < 2:
        return len(response_set) * [0.0]

    tokenizer = _get_tokenizer()
    tokenized_responses = [tokenizer(sentence) for sentence in response_set]
    diversity = []
    for i, hypothesis in enumerate(tokenized_responses):
        other_responses = tokenized_responses[:i] + tokenized_responses[i + 1 :]
        score = sentence_bleu(
            other_responses,
            hypothesis,
            weight,
            smoothing_function=SmoothingFunction().method1,
        )
        diversity.append(1 - score)
    return diversity


def _compute_statistics(results: Dict[str, List[float]]) -> Dict[str, float]:
    statistics: Dict[str, float] = {}
    for metric, values in results.items():
        statistics[f'{metric}_mean'] = float(np.mean(values))
        statistics[f'{metric}_median'] = float(np.median(values))
        statistics[f'{metric}_min'] = float(np.min(values))
        statistics[f'{metric}_max'] = float(np.max(values))
    return statistics


def _get_tokenizer() -> Callable[[str], List[str]]:
    return nltk.word_tokenize


def _download_nltk_resources() -> None:
    logging.info('Downloading NLTK resources.')
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
