import pytest

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.evaluation import ContrastEvaluator


class DummyTask:
    def __init__(self, preference: str):
        self.preference = preference


def dummy_classifier(text: str):
    """Returns a dummy classifier output based on the input text.
    If the text contains 'Chosen', return a score of 0.8.
    If the text contains 'Rejected', return a score of 0.5.
    Otherwise, return a default score of 0.7.
    All outputs use the label "POSITIVE".
    """
    if 'Chosen' in text:
        return [{'label': 'POSITIVE', 'score': 0.8}]
    elif 'Rejected' in text:
        return [{'label': 'POSITIVE', 'score': 0.5}]
    else:
        return [{'label': 'POSITIVE', 'score': 0.7}]


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    """Monkey-patch the pipeline in the module where ContrastEvaluator is defined
    so that it always returns our dummy_classifier.
    """
    monkeypatch.setattr('transformers.pipeline', lambda task: dummy_classifier)


def test_contrast_evaluator_positive():
    """Test ContrastEvaluator.evaluate() with a task preference "positive".
    Using the dummy classifier:
      - For chosen response, dummy returns 0.8.
      - For rejected response, dummy returns 0.5.
    In positive mode, the score is taken as is.
    Contrast = 0.8 - 0.5 = 0.3, which corresponds to 30 (when multiplied by 100).
    """
    task = DummyTask('positive')
    sample = AlignmentDatasetSample(
        prompt='Sample prompt',
        chosen='Chosen text',  # Will yield 0.8
        rejected='Rejected text',  # Will yield 0.5
        id='sample1',
    )
    dataset = AlignmentDataset(task=task, samples=[sample])
    evaluator = ContrastEvaluator()
    scores = evaluator.evaluate(dataset)
    expected = {'sample1': 30}
    assert scores == expected


def test_contrast_evaluator_negative():
    """Test ContrastEvaluator.evaluate() with a task preference "negative".
    In negative mode, since our dummy classifier returns "POSITIVE" for both:
      - For chosen: 1 - 0.8 = 0.2.
      - For rejected: 1 - 0.5 = 0.5.
    Contrast = 0.2 - 0.5 = -0.3, which corresponds to -30.
    """
    task = DummyTask('negative')
    sample = AlignmentDatasetSample(
        prompt='Sample prompt',
        chosen='Chosen text',
        rejected='Rejected text',
        id='sample2',
    )
    dataset = AlignmentDataset(task=task, samples=[sample])
    evaluator = ContrastEvaluator()
    scores = evaluator.evaluate(dataset)
    expected = {'sample2': -30}
    assert scores == expected


def test_contrast_evaluator_polarizing():
    """Test ContrastEvaluator.evaluate() with a task preference "polarizing".
    In polarizing mode, the mode function returns abs(score - 0.5)*2.
      - For chosen: abs(0.8 - 0.5)*2 = 0.6.
      - For rejected: abs(0.5 - 0.5)*2 = 0.
    Contrast = 0.6 - 0 = 0.6, which corresponds to 60.
    """
    task = DummyTask('polarizing')
    sample = AlignmentDatasetSample(
        prompt='Sample prompt',
        chosen='Chosen text',
        rejected='Rejected text',
        id='sample3',
    )
    dataset = AlignmentDataset(task=task, samples=[sample])
    evaluator = ContrastEvaluator()
    scores = evaluator.evaluate(dataset)
    expected = {'sample3': 60}
    assert scores == expected


# --- Test contrast_evaluation() method ---


def test_contrast_evaluation_single_dataset():
    """Test that contrast_evaluation() correctly handles a single AlignmentDataset.
    For a task in positive mode, both samples should yield a contrast of 30.
    """
    task = DummyTask('positive')
    sample1 = AlignmentDatasetSample(
        prompt='Prompt 1',
        chosen='Chosen text',
        rejected='Rejected text',
        id='0',
    )
    sample2 = AlignmentDatasetSample(
        prompt='Prompt 2',
        chosen='Chosen text',
        rejected='Rejected text',
        id='1',
    )
    dataset = AlignmentDataset(task=task, samples=[sample1, sample2])
    evaluator = ContrastEvaluator()
    results = evaluator.contrast_evaluation(dataset)
    expected = [{'0': 30, '1': 30}]
    assert results == expected


def test_contrast_evaluation_continual_dataset():
    """Test that contrast_evaluation() correctly processes a ContinualAlignmentDataset.
    For dataset1 (positive mode), contrast is 30.
    For dataset2 (negative mode), contrast is -30.
    """
    task1 = DummyTask('positive')
    sample1 = AlignmentDatasetSample(
        prompt='Prompt 1',
        chosen='Chosen text',
        rejected='Rejected text',
        id='0',
    )
    dataset1 = AlignmentDataset(task=task1, samples=[sample1])

    task2 = DummyTask('negative')
    sample2 = AlignmentDatasetSample(
        prompt='Prompt 2',
        chosen='Chosen text',
        rejected='Rejected text',
        id='0',
    )
    dataset2 = AlignmentDataset(task=task2, samples=[sample2])

    continual_dataset = ContinualAlignmentDataset(datasets=[dataset1, dataset2])
    evaluator = ContrastEvaluator()
    results = evaluator.contrast_evaluation(continual_dataset)
    expected = [{'0': 30}, {'0': -30}]
    assert results == expected
