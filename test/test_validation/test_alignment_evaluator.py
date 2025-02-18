import pytest

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.dataset.validation.alignment import AlignmentEvaluator


class DummyTask:
    def __init__(self, preference: str = ''):
        self.preference = preference


def dummy_classifier(text):
    """For any given input text, this dummy classifier returns a fixed output.
    In our tests, we use a score of 0.8 with a label of "POSITIVE".
    """
    return [{'label': 'POSITIVE', 'score': 0.8}]


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    """Patch the pipeline in the module where AlignmentEvaluator is defined so that
    it returns our dummy classifier.
    """
    monkeypatch.setattr('transformers.pipeline', lambda task: dummy_classifier)


def test_evaluate_default():
    """Test evaluate() when no task preference is set (defaults to positive mode).
    Expect: For a dummy classifier output {"label": "POSITIVE", "score": 0.8},
    the alignment score is 0.8 * 100 = 80.
    """
    # Create a dummy task with no preference.
    task = DummyTask('')
    sample = AlignmentDatasetSample(
        prompt='Sample prompt', chosen='Some chosen text', rejected='Some rejected text'
    )
    dataset = AlignmentDataset(task=task, samples=[sample])
    evaluator = AlignmentEvaluator()

    # Since the sample has no 'id', the key will be generated from its index ("0")
    scores = evaluator.evaluate(dataset)
    expected = {'0': 80}
    assert scores == expected


def test_evaluate_negative():
    """Test evaluate() when task preference is "negative".
    With a dummy classifier output of {"label": "POSITIVE", "score": 0.8},
    negative mode returns 1 - 0.8 = 0.2, i.e. an expected score of 20.
    """
    task = DummyTask('negative')
    sample = AlignmentDatasetSample(
        prompt='Sample prompt', chosen='Some chosen text', rejected='Some rejected text'
    )
    dataset = AlignmentDataset(task=task, samples=[sample])
    evaluator = AlignmentEvaluator()

    scores = evaluator.evaluate(dataset)
    expected = {'0': 20}
    assert scores == expected


def test_evaluate_polarizing():
    """Test evaluate() when task preference is "polarizing".
    With a dummy classifier output of {"label": "POSITIVE", "score": 0.8},
    polarizing mode returns abs(0.8 - 0.5)*2 = 0.6, i.e. an expected score of 60.
    """
    task = DummyTask('polarizing')
    sample = AlignmentDatasetSample(
        prompt='Sample prompt', chosen='Some chosen text', rejected='Some rejected text'
    )
    dataset = AlignmentDataset(task=task, samples=[sample])
    evaluator = AlignmentEvaluator()

    scores = evaluator.evaluate(dataset)
    expected = {'0': 60}
    assert scores == expected


def test_alignment_evaluation_single_dataset():
    """Test that alignment_evaluation() returns a one-element list containing
    the evaluation dictionary for a single AlignmentDataset.
    """
    task = DummyTask(
        'positive'
    )  # even if explicitly positive, default behavior is similar.
    sample1 = AlignmentDatasetSample(
        prompt='Prompt 1', chosen='Chosen text 1', rejected='Rejected text'
    )
    sample2 = AlignmentDatasetSample(
        prompt='Prompt 2', chosen='Chosen text 2', rejected='Rejected text'
    )
    dataset = AlignmentDataset(task=task, samples=[sample1, sample2])
    evaluator = AlignmentEvaluator()

    results = evaluator.alignment_evaluation(dataset)
    # Expect keys "0" and "1" with scores based on positive mode, i.e. 80.
    expected = [{'0': 80, '1': 80}]
    assert results == expected


def test_alignment_evaluation_continual_dataset():
    """Test that alignment_evaluation() handles a ContinualAlignmentDataset correctly,
    returning a list of evaluation dictionaries (one per sub-dataset).
    """
    task1 = DummyTask('positive')
    sample1 = AlignmentDatasetSample(
        prompt='Prompt 1', chosen='Chosen text 1', rejected='Rejected text'
    )
    dataset1 = AlignmentDataset(task=task1, samples=[sample1])

    task2 = DummyTask('negative')
    sample2 = AlignmentDatasetSample(
        prompt='Prompt 2', chosen='Chosen text 2', rejected='Rejected text'
    )
    dataset2 = AlignmentDataset(task=task2, samples=[sample2])

    continual_dataset = ContinualAlignmentDataset(datasets=[dataset1, dataset2])
    evaluator = AlignmentEvaluator()

    results = evaluator.alignment_evaluation(continual_dataset)
    # In dataset1: default positive mode returns 80.
    # In dataset2: negative mode returns 1 - 0.8 = 0.2, i.e. 20.
    expected = [{'0': 80}, {'0': 20}]
    assert results == expected
