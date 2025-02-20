from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample
from aif_gen.dataset.validation.alignment import AlignmentEvaluator


class DummyJudge:
    """A dummy judge to replace the real LLM pipeline.
    It returns controlled outputs from a list of provided outputs.
    """

    def __init__(self, outputs):
        self.outputs = outputs
        self.index = 0

    def __call__(self, prompt, max_length, do_sample):
        output = self.outputs[self.index]
        self.index += 1
        return [{'generated_text': output}]


def test_parse_rating_valid():
    evaluator = AlignmentEvaluator()
    text = 'The rating is 0.75 out of 1.'
    rating = evaluator._parse_rating(text)
    assert rating == 0.75, 'Should extract a valid rating from the text.'


def test_parse_rating_invalid():
    evaluator = AlignmentEvaluator()
    text = 'No valid number here.'
    rating = evaluator._parse_rating(text)
    assert rating == 0.5, 'Should return fallback 0.5 when no number is found.'


def test_parse_rating_clamps():
    evaluator = AlignmentEvaluator()
    # test clamping below 0.
    rating_low = evaluator._parse_rating('Rating: -0.3')
    assert rating_low == 0.0, 'Negative ratings should be clamped to 0.0'
    # test clamping above 1.
    rating_high = evaluator._parse_rating('Rating: 1.3')
    assert rating_high == 1.0, 'Ratings above 1 should be clamped to 1.0'


def test_evaluate_with_failure():
    """Tests evaluate() when one sample returns a valid rating and another fails to include a number.
    The evaluator should record a failure rate accordingly.
    """
    samples = [
        AlignmentDatasetSample('Prompt 1', 'Chosen 1', 'Rejected 1'),
        AlignmentDatasetSample('Prompt 2', 'Chosen 2', 'Rejected 2'),
    ]
    dataset = AlignmentDataset(task=None, samples=samples)

    # first output contains "0.8" (valid), second output does not contain any number.
    dummy_outputs = ['The rating is 0.8', 'No number provided']
    evaluator = AlignmentEvaluator()
    evaluator.judge = DummyJudge(dummy_outputs)

    scores = evaluator.evaluate(dataset)
    # expected scores: 0.8 and fallback 0.5.
    assert scores == [
        0.8,
        0.5,
    ], "Scores should match the dummy outputs' parsed ratings."
    assert (
        evaluator.failure_rate == 0.5
    ), 'Failure rate should be 0.5 (1 failure out of 2 samples).'


def test_evaluate_without_failure():
    """Tests evaluate() when the dummy outputs explicitly include the string '0.5',
    so that even though the rating is 0.5, it is not considered a failure.
    """
    samples = [
        AlignmentDatasetSample('Prompt 1', 'Chosen 1', 'Rejected 1'),
        AlignmentDatasetSample('Prompt 2', 'Chosen 2', 'Rejected 2'),
    ]
    dataset = AlignmentDataset(task=None, samples=samples)

    dummy_outputs = ['The rating is 0.5', 'Also, rating: 0.5']
    evaluator = AlignmentEvaluator()
    evaluator.judge = DummyJudge(dummy_outputs)

    scores = evaluator.evaluate(dataset)
    assert scores == [0.5, 0.5], 'Both samples should have a score of 0.5.'
    assert (
        evaluator.failure_rate == 0.0
    ), 'There should be no failures when fallback is explicitly provided.'
