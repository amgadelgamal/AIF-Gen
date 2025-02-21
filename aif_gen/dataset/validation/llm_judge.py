import logging
import re
from typing import Dict, List, Optional

import numpy as np
import transformers
from transformers import pipeline

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset


def llm_judge_validation(dataset: Dataset) -> List[Optional[Dict[str, float]]]:
    r"""Use an LLM to judge the quality of the dataset.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.

    Returns:
        List[Optional[Dict[str, float]]]: For every AlignmentDataset, returns a dictionary with entries of the form '{metric}_{stat}':
            - Stat is one of ['mean', 'median', 'min', 'max']
            - Metric is one of:
                'alignment_chosen'    -> The alignment between the chosen response and prompt, as determined by the LLM.
                'alignment_rejected'  -> The alignment between the rejected response and prompt, as determined by the LLM.
                'coherence_chosen'    -> The coherence in the chosen response, as determined by the LLM.
                'coherence_rejected'  -> The coherence in the rejected response, as determined by the LLM.

    Note:
        - If the dataset is empty, we put None in place of the dictionary.
    """
    if isinstance(dataset, AlignmentDataset):
        datasets = [dataset]
    else:
        # This assert is here to make mypy happy
        assert isinstance(dataset, ContinualAlignmentDataset)
        datasets = dataset.datasets

    results = []
    judge = _init_llm_judge(model_name='gpt2')
    for dataset in datasets:
        if len(dataset):
            result = _llm_judge_validation(dataset, judge)
        else:
            logging.warning(f'Skipping LLM judge on empty dataset: {dataset}')
            result = None
        results.append(result)
    return results


def _init_llm_judge(model_name: str) -> transformers.Pipeline:
    logging.debug(f'Running LLM judge validation with model: {model_name}')
    return pipeline(
        'text-generation',
        model=model_name,
        tokenizer=model_name,
        max_new_tokens=32,
        do_sample=False,
        truncation=True,
        pad_token_id=50256,
        return_full_text=False,
    )


def _llm_judge_validation(
    dataset: AlignmentDataset, judge: transformers.Pipeline
) -> Dict[str, float]:
    results: Dict[str, List[float]] = {
        'alignment_chosen': [],
        'alignment_rejected': [],
        'coherence_chosen': [],
        'coherence_rejected': [],
    }

    for sample in dataset.samples:
        score = _judge_alignment(judge, sample.prompt, sample.chosen)
        if score is not None:
            results['alignment_chosen'].append(score)

        score = _judge_alignment(judge, sample.prompt, sample.rejected)
        if score is not None:
            results['alignment_rejected'].append(score)

        score = _judge_coherence(judge, sample.chosen)
        if score is not None:
            results['coherence_chosen'].append(score)

        score = _judge_coherence(judge, sample.rejected)
        if score is not None:
            results['coherence_rejected'].append(score)

    inferences = 4 * len(dataset)
    failures = inferences - sum([len(metric) for metric in results.values()])
    if failures > 0:
        logging.warning(f'LLM judge parsing failures: {failures} / {4 * len(dataset)}')
    if failures == inferences:
        raise RuntimeError('Could not parse LLM judge output for any samples')

    return _compute_statistics(results)


def _judge_alignment(
    judge: transformers.Pipeline, prompt: str, response: str
) -> Optional[float]:
    judge_prompt = (
        'Please evaluate the alignment of the following response with respect to the given prompt. '
        'Consider whether the response aligns with the prompt in terms of clarity, relevance, and correctness. '
        'Provide a rating from 0 to 1 where 1 indicates perfect alignment and 0 indicates no alignment at all.\n\n'
        f'Prompt: {prompt}\n\n'
        f'Response: {response}\n\n'
        'Alignment Score (0 to 1):'
    )
    judge_output = judge(judge_prompt)[0]['generated_text']
    return _parse_judge_output(judge_output)


def _judge_coherence(judge: transformers.Pipeline, response: str) -> Optional[float]:
    judge_prompt = (
        'Please evaluate the coherence of the following response on a scale from 0 to 1, '
        'where 1 indicates excellent coherence and 0 indicates poor coherence:\n\n'
        f'Response: {response}\n\n'
        'Coherence Score (0 to 1):'
    )
    judge_output = judge(judge_prompt)[0]['generated_text']
    return _parse_judge_output(judge_output)


def _parse_judge_output(judge_output: str) -> Optional[float]:
    logging.debug(f'Parsing judge output rating: {judge_output}')
    match = re.search(r'([-+]?[0-9]*\.?[0-9]+)', judge_output)
    try:
        assert match is not None
        rating = float(match.group(1))
        return max(0, min(1, rating))
    except Exception:
        logging.warning(f'Failed to parse judge output rating: {judge_output}')
        return None


def _compute_statistics(results: Dict[str, List[float]]) -> Dict[str, float]:
    statistics: Dict[str, float] = {}
    for metric, values in results.items():
        statistics[f'{metric}_mean'] = float(np.mean(values))
        statistics[f'{metric}_median'] = float(np.median(values))
        statistics[f'{metric}_min'] = float(np.min(values))
        statistics[f'{metric}_max'] = float(np.max(values))
    return statistics
