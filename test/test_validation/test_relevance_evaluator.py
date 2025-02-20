# Dummy classes to simulate AlignmentDataset and its sample objects.
class DummySample:
    def __init__(self, prompt: str, chosen: str):
        self.prompt = prompt
        self.chosen = chosen


class DummyAlignmentDataset:
    def __init__(self, samples):
        self.samples = samples


# A dummy judge function that simulates the output of the text-generation pipeline.
def dummy_judge(prompt, max_length=50, do_sample=False):
    # This dummy always returns a rating of 0.9 regardless of the prompt.
    # The evaluator will use a regular expression to parse this number.
    return [{'generated_text': ' 0.9'}]


def test_relevance_evaluator(monkeypatch):
    from aif_gen.dataset.validation.relevance import RelevanceEvaluator

    # Create a dummy dataset with two samples.
    samples = [
        DummySample(
            prompt='What is the capital of France?',
            chosen='Paris is the capital of France.',
        ),
        DummySample(
            prompt='Who is the president of the USA?',
            chosen='Joe Biden is the current president.',
        ),
    ]
    dataset = DummyAlignmentDataset(samples)

    evaluator = RelevanceEvaluator()
    monkeypatch.setattr(evaluator, 'judge', dummy_judge)

    # Run the evaluator.
    scores = evaluator.evaluate(dataset)

    # Verify that we get a score of 0.9 for each sample.
    assert len(scores) == len(samples)
    for score in scores:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == 0.9
