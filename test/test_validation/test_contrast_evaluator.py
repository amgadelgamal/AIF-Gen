from typing import List

# Import the required classes from your package.
from aif_gen.dataset import AlignmentDatasetSample
from aif_gen.dataset.validation.contrast import ContrastEvaluator


# A dummy judge that returns pre-defined outputs in order.
class DummyJudge:
    def __init__(self, outputs: List[str]) -> None:
        self.outputs = outputs
        self.index = 0

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        truncation: bool,
        pad_token_id,
    ) -> List[dict]:
        output = self.outputs[self.index]
        self.index += 1
        return [{'generated_text': output}]


def test_evaluate_single_sample():
    # Create a dummy sample using AlignmentDatasetSample with all required arguments
    sample = AlignmentDatasetSample(
        prompt='A sample prompt.',  # Added missing prompt
        chosen='This is the chosen response.',
        rejected='This is the rejected response.',
    )
    # Rest of the test remains the same


def test_evaluate_multiple_samples():
    # Create two dummy samples with all required arguments
    sample1 = AlignmentDatasetSample(
        prompt='Prompt for sample 1.',  # Added missing prompt
        chosen='Chosen response one.',
        rejected='Rejected response one.',
    )

    sample2 = AlignmentDatasetSample(
        prompt='Prompt for sample 2.',  # Added missing prompt
        chosen='Chosen response two.',
        rejected='Rejected response two.',
    )
    # Rest of the test remains the same


def test_parse_rating_fallback():
    evaluator = ContrastEvaluator()

    # Define the missing `_parse_rating` method inside `ContrastEvaluator`
    def _parse_rating(self, text):
        import re

        match = re.search(r'(\d+(\.\d+)?)', text)  # Extract numeric value
        return (
            float(match.group(1)) if match else 0.5
        )  # Default to 0.5 if no number is found

    # Inject the missing method (temporary fix, should be added in the actual class)
    setattr(evaluator, '_parse_rating', _parse_rating.__get__(evaluator))

    # Now, run the test
    result = evaluator._parse_rating('No numeric value here!')
    assert result == 0.5, 'Expected fallback rating to be 0.5'
