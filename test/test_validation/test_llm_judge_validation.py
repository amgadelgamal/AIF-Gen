import pytest

from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample


class DummyJudge:
    def __init__(self, outputs):
        self.outputs = outputs
        self.index = 0

    def __call__(self, prompt, max_new_tokens, do_sample, truncation, pad_token_id):
        output = self.outputs[self.index]
        self.index += 1
        return [{'generated_text': output}]


@pytest.mark.skip('TODO: Implement llm_judge_validation_tests')
def test_parse_rating_valid():
    text = 'The rating is 0.75 out of 1.'
    rating = evaluator._parse_rating(text)
    assert rating == 0.75, 'Should extract a valid rating from the text.'


@pytest.mark.skip('TODO: Implement llm_judge_validation_tests')
def test_parse_rating_invalid():
    text = 'No valid number here.'
    rating = evaluator._parse_rating(text)
    assert rating == 0.5, 'Should return fallback 0.5 when no number is found.'


@pytest.mark.skip('TODO: Implement llm_judge_validation_tests')
def test_parse_rating_clamps():
    rating_low = evaluator._parse_rating('Rating: -0.3')
    assert rating_low == 0.0, 'Negative ratings should be clamped to 0.0'
    rating_high = evaluator._parse_rating('Rating: 1.3')
    assert rating_high == 1.0, 'Ratings above 1 should be clamped to 1.0'


@pytest.mark.skip('TODO: Implement llm_judge_validation_tests')
def test_evaluate_with_failure():
    samples = [
        AlignmentDatasetSample('Prompt 1', 'Chosen 1', 'Rejected 1'),
        AlignmentDatasetSample('Prompt 2', 'Chosen 2', 'Rejected 2'),
    ]
    dataset = AlignmentDataset(task=None, samples=samples)

    # first output contains "0.8" (valid), second output does not contain any number.
    dummy_outputs = ['The rating is 0.8', 'No number provided']
    evaluator.judge = DummyJudge(dummy_outputs)

    scores = evaluator.evaluate(dataset)
    # expected scores: 0.8 and fallback 0.5.
    assert scores == [0.8, 0.5]
    assert evaluator.failure_rate == 0.5


@pytest.mark.skip('TODO: Implement llm_judge_validation_tests')
def test_evaluate_without_failure():
    samples = [
        AlignmentDatasetSample('Prompt 1', 'Chosen 1', 'Rejected 1'),
        AlignmentDatasetSample('Prompt 2', 'Chosen 2', 'Rejected 2'),
    ]
    dataset = AlignmentDataset(task=None, samples=samples)

    dummy_outputs = ['The rating is 0.5', 'Also, rating: 0.5']
    evaluator.judge = DummyJudge(dummy_outputs)

    scores = evaluator.evaluate(dataset)
    assert scores == [0.5, 0.5], 'Both samples should have a score of 0.5.'
    assert evaluator.failure_rate == 0.0
