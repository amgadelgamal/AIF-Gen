from typing import List

# Import necessary components from your package.
from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample
from aif_gen.dataset.validation.coherence import CoherenceEvaluator


class DummyJudge:
    """A dummy judge to replace the LLM pipeline.
    It returns pre-defined outputs (as a list of strings) in order.
    """

    def __init__(self, outputs: List[str]) -> None:
        self.outputs = outputs
        self.index = 0

    def __call__(self, prompt: str, max_length: int, do_sample: bool) -> List[dict]:
        output = self.outputs[self.index]
        self.index += 1
        return [{'generated_text': output}]


def test_parse_rating_valid():
    evaluator = CoherenceEvaluator()
    # Test _parse_rating with a string containing a valid score.
    text = 'The coherence score is 0.85 out of 1.'
    rating = evaluator._parse_rating(text)
    assert abs(rating - 0.85) < 1e-6, 'Should correctly extract and parse the score.'


def test_parse_rating_fallback():
    evaluator = CoherenceEvaluator()
    # When no number is found, _parse_rating should return 0.5.
    text = 'No numeric value available.'
    rating = evaluator._parse_rating(text)
    assert (
        abs(rating - 0.5) < 1e-6
    ), 'Should return fallback score of 0.5 when no number is present.'


def test_evaluate_single_sample():
    # Create a dummy sample; only the 'chosen' field is used for coherence evaluation.
    sample = AlignmentDatasetSample(
        chosen='This is a coherent chosen response.',
        rejected='This field is ignored by the coherence evaluator.',
    )
    dataset = AlignmentDataset(task=None, samples=[sample])

    # Dummy output for the chosen response; the output contains a score of 0.75.
    dummy_outputs = ['Score: 0.75']

    evaluator = CoherenceEvaluator()
    evaluator.judge = DummyJudge(dummy_outputs)

    scores = evaluator.evaluate(dataset)
    assert len(scores) == 1, 'Should return one score for one sample.'
    assert abs(scores[0] - 0.75) < 1e-6, 'The extracted coherence score should be 0.75.'


def test_evaluate_multiple_samples():
    # Create two dummy samples.
    sample1 = AlignmentDatasetSample(
        chosen='Chosen response one for coherence test.',
        rejected='Irrelevant for coherence.',
    )
    sample2 = AlignmentDatasetSample(
        chosen='Chosen response two with different coherence.',
        rejected='Irrelevant for coherence.',
    )
    dataset = AlignmentDataset(task=None, samples=[sample1, sample2])

    # Provide dummy outputs for each sample (one for each chosen response).
    dummy_outputs = [
        'Coherence Score: 0.90',  # For sample1
        'Coherence Score: 0.60',  # For sample2
    ]

    evaluator = CoherenceEvaluator()
    evaluator.judge = DummyJudge(dummy_outputs)

    scores = evaluator.evaluate(dataset)
    expected_scores = [0.90, 0.60]
    assert len(scores) == 2, 'Should return two scores for two samples.'
    for score, expected in zip(scores, expected_scores):
        assert (
            abs(score - expected) < 1e-6
        ), 'Each score should match the expected value.'
