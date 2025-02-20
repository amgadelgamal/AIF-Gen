import numpy as np
import pytest

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.dataset.validation import diversity_validation


def test_diversity_validation():
    samples = [
        AlignmentDatasetSample(
            'Mock prompt A 1', 'Winning Response A 1', 'Losing Response A 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt B 1', 'Winning Response B 1', 'Losing Response B 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt C 1', 'Winning Response C 1', 'Losing Response C 1'
        ),
    ]
    dataset = AlignmentDataset(task=None, samples=samples)

    result = diversity_validation(dataset)
    exp_result = [
        {
            'chosen_diversity_max': np.float64(0.767920558319361),
            'chosen_diversity_mean': np.float64(0.767920558319361),
            'chosen_diversity_median': np.float64(0.767920558319361),
            'chosen_diversity_min': np.float64(0.767920558319361),
            'prompt_diversity_max': np.float64(0.767920558319361),
            'prompt_diversity_mean': np.float64(0.767920558319361),
            'prompt_diversity_median': np.float64(0.767920558319361),
            'prompt_diversity_min': np.float64(0.767920558319361),
            'rejected_diversity_max': np.float64(0.767920558319361),
            'rejected_diversity_mean': np.float64(0.767920558319361),
            'rejected_diversity_median': np.float64(0.767920558319361),
            'rejected_diversity_min': np.float64(0.767920558319361),
        }
    ]
    assert result == exp_result


def test_diversity_validation_identical_prompts():
    samples = [
        AlignmentDatasetSample(
            'Mock prompt', 'Winning Response A 1', 'Losing Response A 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt', 'Winning Response B 1', 'Losing Response B 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt', 'Winning Response C 1', 'Losing Response C 1'
        ),
    ]
    dataset = AlignmentDataset(task=None, samples=samples)

    result = diversity_validation(dataset)
    exp_result = [
        {
            'chosen_diversity_max': np.float64(0.767920558319361),
            'chosen_diversity_mean': np.float64(0.767920558319361),
            'chosen_diversity_median': np.float64(0.767920558319361),
            'chosen_diversity_min': np.float64(0.767920558319361),
            'prompt_diversity_max': np.float64(0.535841116638722),
            'prompt_diversity_mean': np.float64(0.535841116638722),
            'prompt_diversity_median': np.float64(0.535841116638722),
            'prompt_diversity_min': np.float64(0.535841116638722),
            'rejected_diversity_max': np.float64(0.767920558319361),
            'rejected_diversity_mean': np.float64(0.767920558319361),
            'rejected_diversity_median': np.float64(0.767920558319361),
            'rejected_diversity_min': np.float64(0.767920558319361),
        }
    ]
    assert result == exp_result


@pytest.mark.parametrize('ngram', [1, 2, 3])
def test_diversity_validation_single_sample_dataset(ngram):
    samples = [
        AlignmentDatasetSample(
            'Mock prompt A', 'Winning Response A 1', 'Losing Response A 1'
        ),
    ]
    dataset = AlignmentDataset(task=None, samples=samples)

    result = diversity_validation(dataset, ngram)

    exp_result = [
        {
            'chosen_diversity_max': 0.0,
            'chosen_diversity_mean': 0.0,
            'chosen_diversity_median': 0.0,
            'chosen_diversity_min': 0.0,
            'prompt_diversity_max': 0.0,
            'prompt_diversity_mean': 0.0,
            'prompt_diversity_median': 0.0,
            'prompt_diversity_min': 0.0,
            'rejected_diversity_max': 0.0,
            'rejected_diversity_mean': 0.0,
            'rejected_diversity_median': 0.0,
            'rejected_diversity_min': 0.0,
        }
    ]
    assert result == exp_result


@pytest.mark.parametrize('ngram', [1, 2, 3])
def test_diversity_validation_empty_dataset(ngram):
    samples = []
    dataset = AlignmentDataset(task=None, samples=samples)

    result = diversity_validation(dataset, ngram=ngram)
    assert result == [None]


@pytest.mark.parametrize('bad_ngram', [-1, None, 0, 'foo'])
def test_diversity_validation_bad_ngram(bad_ngram):
    dataset = AlignmentDataset(task=None, samples=[])

    with pytest.raises(ValueError):
        diversity_validation(dataset, ngram=bad_ngram)


def test_diversity_validation_continual_dataset():
    samples = [
        AlignmentDatasetSample(
            'Mock prompt', 'Winning Response A 1', 'Losing Response A 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt', 'Winning Response B 1', 'Losing Response B 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt', 'Winning Response C 1', 'Losing Response C 1'
        ),
    ]
    dataset1 = AlignmentDataset(task=None, samples=samples)
    dataset2 = AlignmentDataset(task=None, samples=[])
    dataset = ContinualAlignmentDataset(datasets=[dataset1, dataset2])

    result = diversity_validation(dataset)
    exp_result = [
        {
            'chosen_diversity_max': np.float64(0.767920558319361),
            'chosen_diversity_mean': np.float64(0.767920558319361),
            'chosen_diversity_median': np.float64(0.767920558319361),
            'chosen_diversity_min': np.float64(0.767920558319361),
            'prompt_diversity_max': np.float64(0.535841116638722),
            'prompt_diversity_mean': np.float64(0.535841116638722),
            'prompt_diversity_median': np.float64(0.535841116638722),
            'prompt_diversity_min': np.float64(0.535841116638722),
            'rejected_diversity_max': np.float64(0.767920558319361),
            'rejected_diversity_mean': np.float64(0.767920558319361),
            'rejected_diversity_median': np.float64(0.767920558319361),
            'rejected_diversity_min': np.float64(0.767920558319361),
        },
        None,
    ]
    assert result == exp_result
