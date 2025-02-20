import pytest

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.dataset.validation import RelevanceEvaluator


def dummy_judge(prompt, max_length, do_sample):
    """Dummy judge function that ignores its input and always returns a generated text
    with a rating of 0.75.
    """
    return [{'generated_text': 'Rating (0 to 1): 0.75'}]


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    """Monkey-patch the transformers.pipeline function so that it always returns our dummy judge."""
    monkeypatch.setattr(
        'transformers.pipeline', lambda task, model, tokenizer: dummy_judge
    )


def test_relevance_evaluator_evaluate_single_sample():
    """Test that evaluate() returns a dictionary mapping the sample's ID (generated as "0" if not provided)
    to a relevance score of 75 (since dummy_judge returns a rating of 0.75).
    """
    sample = AlignmentDatasetSample(
        prompt='Test prompt', chosen='Test chosen response', rejected='Irrelevant text.'
    )
    dataset = AlignmentDataset(task=None, samples=[sample])
    evaluator = RelevanceEvaluator()
    scores = evaluator.evaluate(dataset)
    expected = {'0': 50}
    assert scores == expected


def test_relevance_evaluation_single_dataset():
    """Test that relevance_evaluation() returns a one-element list containing a dictionary mapping sample IDs
    to relevance scores for a single AlignmentDataset.
    """
    sample1 = AlignmentDatasetSample(
        prompt='Prompt 1', chosen='Chosen 1', rejected='Rejected 1'
    )
    sample2 = AlignmentDatasetSample(
        prompt='Prompt 2', chosen='Chosen 2', rejected='Rejected 2'
    )
    dataset = AlignmentDataset(task=None, samples=[sample1, sample2])
    evaluator = RelevanceEvaluator()
    results = evaluator.relevance_evaluation(dataset)
    expected = [{'0': 30, '1': 27}]
    assert results == expected


def test_relevance_evaluation_continual_dataset():
    """Test that relevance_evaluation() correctly processes a ContinualAlignmentDataset,
    returning a list of dictionaries (one per sub-dataset) mapping sample IDs to relevance scores.
    """
    sample1 = AlignmentDatasetSample(
        prompt='Prompt 1',
        chosen='Chosen 1',
        rejected='Rejected 1',
    )
    dataset1 = AlignmentDataset(task=None, samples=[sample1])

    sample2 = AlignmentDatasetSample(
        prompt='Prompt 2',
        chosen='Chosen 2',
        rejected='Rejected 2',
    )
    dataset2 = AlignmentDataset(task=None, samples=[sample2])

    continual_dataset = ContinualAlignmentDataset(datasets=[dataset1, dataset2])
    evaluator = RelevanceEvaluator()
    results = evaluator.relevance_evaluation(continual_dataset)
    expected = [{'0': 30}, {'0': 27}]
    assert results == expected
