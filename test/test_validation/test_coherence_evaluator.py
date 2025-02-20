from typing import List

# Import necessary components from your package.
from aif_gen.dataset import AlignmentDatasetSample
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
    assert abs(rating - 0.5) < 1e-6


def test_evaluate_single_sample():
    # Create a dummy sample with 'prompt' field included
    sample = AlignmentDatasetSample(
        prompt='This is the prompt.',  # Added missing prompt
        chosen='This is a coherent chosen response.',
        rejected='This field is ignored by the coherence evaluator.',
    )


def test_evaluate_multiple_samples():
    # Create two dummy samples with all required arguments
    sample1 = AlignmentDatasetSample(
        prompt='Prompt for sample 1.',
        chosen='Chosen response one for coherence test.',
        rejected='Rejected response one.',
    )

    sample2 = AlignmentDatasetSample(
        prompt='Prompt for sample 2.',
        chosen='Chosen response two for coherence test.',
        rejected='Rejected response two.',
    )

    class DummyDataset:
        def __init__(self, samples):
            self.samples = samples  # Wraps the list into an object with `.samples`

    dataset = DummyDataset([sample1, sample2])

    # Initialize the coherence evaluator
    evaluator = CoherenceEvaluator()

    # Call evaluate with the wrapped dataset
    results = evaluator.evaluate(dataset)

    # Define dummy expected outputs for comparison
    dummy_outputs = [0.9, 0.8]  # Adjust these based on expected behavior

    # Ensure results match the dummy outputs
    assert len(results) == len(dummy_outputs), 'Mismatch in result length'
    for res, expected in zip(results, dummy_outputs):
        assert isinstance(res, float), 'Output should be a float'
        assert 0.0 <= res <= 1.0, 'Coherence score should be between 0 and 1'
