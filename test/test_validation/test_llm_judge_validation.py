import pytest

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.dataset.validation.llm_judge import llm_judge_validation


@pytest.mark.parametrize('judge_rating', [-1, -1.23, -0.23, 0.12, 0.23, 0.5, 1.23, 10])
@pytest.mark.skip('TODO')
def test_llm_judge_validation(mocker, judge_rating):
    mocker.patch(
        'aif_gen.dataset.validation.llm_judge._init_llm_judge',
        return_value=lambda _: [{'generated_text': f'LLM Response: {judge_rating}'}],
    )

    samples = [
        AlignmentDatasetSample(
            'Mock prompt A', 'Winning Response A 1', 'Losing Response A 1'
        ),
    ]
    dataset = AlignmentDataset(task=None, samples=samples)
    result = llm_judge_validation(dataset)
    exp_keys = [
        'alignment_chosen_max',
        'alignment_chosen_mean',
        'alignment_chosen_median',
        'alignment_chosen_min',
        'alignment_rejected_max',
        'alignment_rejected_mean',
        'alignment_rejected_median',
        'alignment_rejected_min',
        'coherence_chosen_max',
        'coherence_chosen_mean',
        'coherence_chosen_median',
        'coherence_chosen_min',
        'coherence_rejected_max',
        'coherence_rejected_mean',
        'coherence_rejected_median',
        'coherence_rejected_min',
    ]
    assert isinstance(result, list)
    assert sorted(list(result[0].keys())) == sorted(exp_keys)
    for v in result[0].values():
        assert v == max(0.0, min(1.0, judge_rating))


@pytest.mark.skip('TODO')
def test_llm_judge_validation_empty_dataset(mocker):
    mocker.patch(
        'aif_gen.dataset.validation.llm_judge._init_llm_judge',
        return_value=lambda _: f'LLM Response: 0.1234',
    )

    dataset = AlignmentDataset(task=None, samples=[])
    result = llm_judge_validation(dataset)
    assert result == [None]


@pytest.mark.parametrize('judge_rating', [-1, -1.23, -0.23, 0.12, 0.23, 0.5, 1.23, 10])
@pytest.mark.skip('TODO')
def test_llm_judge_validation_continual_dataset(mocker, judge_rating):
    mocker.patch(
        'aif_gen.dataset.validation.llm_judge._init_llm_judge',
        return_value=lambda _: [{'generated_text': f'LLM response {judge_rating}'}],
    )

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

    result = llm_judge_validation(dataset)

    exp_keys = [
        'alignment_chosen_max',
        'alignment_chosen_mean',
        'alignment_chosen_median',
        'alignment_chosen_min',
        'alignment_rejected_max',
        'alignment_rejected_mean',
        'alignment_rejected_median',
        'alignment_rejected_min',
        'coherence_chosen_max',
        'coherence_chosen_mean',
        'coherence_chosen_median',
        'coherence_chosen_min',
        'coherence_rejected_max',
        'coherence_rejected_mean',
        'coherence_rejected_median',
        'coherence_rejected_min',
    ]
    assert isinstance(result, list)
    assert sorted(list(result[0].keys())) == sorted(exp_keys)
    for v in result[0].values():
        assert v == max(0.0, min(1.0, judge_rating))
    assert result[1] == None


@pytest.mark.skip('TODO')
def test_diversity_validation_all_parses_failed(mocker):
    mocker.patch(
        'aif_gen.dataset.validation.llm_judge._init_llm_judge',
        return_value=lambda _: [{'generated_text': f'Bad LLM response with no rating'}],
    )

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
    with pytest.raises(RuntimeError):
        llm_judge_validation(dataset)
