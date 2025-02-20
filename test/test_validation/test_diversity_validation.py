import pytest

from aif_gen.dataset import AlignmentDataset
from aif_gen.dataset.validation.diversity import DiversityEvaluator


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for testing."""

    class Sample:
        def __init__(self, chosen):
            self.chosen = (
                chosen  # Only 'chosen' responses are used for diversity calculation
            )

    samples = [
        Sample('The quick brown fox jumps over the lazy dog.'),
        Sample('A completely different response to ensure variation.'),
        Sample('Another response to make the dataset diverse.'),
    ]

    return AlignmentDataset(samples=samples, task='dummy_task')


def test_compute_response_diversity():
    """Test the diversity computation on a varied response set."""
    evaluator = DiversityEvaluator(ngram=3)

    response_set = [
        'The quick brown fox jumps over the lazy dog.',
        'A different sentence that does not match.',
        'Yet another distinct phrase with unique words.',
    ]

    diversity_score = evaluator.compute_response_diversity(response_set)

    assert isinstance(diversity_score, float), 'Diversity score should be a float'
    assert 0.0 <= diversity_score <= 1.0, 'Diversity score should be between 0 and 1'


def test_compute_response_diversity_identical_responses():
    """Test diversity computation when all responses are identical (should return low diversity)."""
    evaluator = DiversityEvaluator()

    response_set = [
        'Repeated sentence.',
        'Repeated sentence.',
        'Repeated sentence.',
    ]

    diversity_score = evaluator.compute_response_diversity(response_set)
    assert diversity_score == pytest.approx(0.0, abs=1e-5)


def test_compute_response_diversity_single_response():
    """Test edge case where only one response is given (should return 0.0)."""
    evaluator = DiversityEvaluator()

    response_set = ['Only one response.']
    diversity_score = evaluator.compute_response_diversity(response_set)
    assert diversity_score == 0.0


def test_evaluate(dummy_dataset):
    """Test that the evaluate method correctly applies the diversity metric."""
    evaluator = DiversityEvaluator()
    scores = evaluator.evaluate(dummy_dataset)

    assert isinstance(scores, list), 'evaluate() should return a list'
    assert all(isinstance(score, float) for score in scores)
    assert len(scores) == len(dummy_dataset.samples)


def test_evaluate_empty_dataset():
    """Test evaluate() on an empty dataset (should return empty list)."""
    empty_dataset = AlignmentDataset(samples=[], task='empty_task')

    evaluator = DiversityEvaluator()
    scores = evaluator.evaluate(empty_dataset)

    assert scores == [], 'Empty dataset should return an empty list'
