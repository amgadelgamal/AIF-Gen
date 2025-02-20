# from typing import List

# # Import the required classes from your package.
# from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample
# from aif_gen.dataset.validation.contrast import ContrastEvaluator


# # A dummy judge that returns pre-defined outputs in order.
# class DummyJudge:
#     def __init__(self, outputs: List[str]) -> None:
#         self.outputs = outputs
#         self.index = 0

#     def __call__(self, prompt: str, max_length: int, do_sample: bool) -> List[dict]:
#         output = self.outputs[self.index]
#         self.index += 1
#         return [{'generated_text': output}]


# def test_parse_rating_fallback():
#     evaluator = ContrastEvaluator()
#     # When no number is found in the text, _parse_rating should return 0.5.
#     result = evaluator._parse_rating('No numeric value here!')
#     assert result == 0.5, 'Fallback rating should be 0.5 when no number is present.'


# def test_evaluate_single_sample():
#     # Create a dummy sample using AlignmentDatasetSample.
#     sample = AlignmentDatasetSample(
#         'This is the chosen response.', 'This is the rejected response.'
#     )
#     dataset = AlignmentDataset(task=None, samples=[sample])

#     # Dummy outputs for one sample (2 calls: chosen then rejected).
#     # For example, chosen returns text containing "0.8" and rejected returns "0.3".
#     dummy_outputs = [
#         'Generated output with score 0.8',  # for chosen response
#         'Generated output with score 0.3',  # for rejected response
#     ]

#     evaluator = ContrastEvaluator()
#     evaluator.judge = DummyJudge(dummy_outputs)

#     scores = evaluator.evaluate(dataset)

#     # Expected contrast: 0.8 - 0.3 = 0.5.
#     expected_contrast = 0.8 - 0.3
#     assert len(scores) == 1
#     assert (
#         abs(scores[0] - expected_contrast) < 1e-6
#     ), 'Contrast score should be the difference of the ratings.'


# def test_evaluate_multiple_samples():
#     # Create two dummy samples.
#     sample1 = AlignmentDatasetSample('Chosen response one.', 'Rejected response one.')
#     sample2 = AlignmentDatasetSample('Chosen response two.', 'Rejected response two.')
#     dataset = AlignmentDataset(task=None, samples=[sample1, sample2])

#     # Provide dummy outputs for each sample (4 calls total).
#     # For sample 1: chosen -> "0.9", rejected -> "0.4" (contrast = 0.5)
#     # For sample 2: chosen -> "0.7", rejected -> "0.2" (contrast = 0.5)
#     dummy_outputs = ['Score: 0.9', 'Score: 0.4', 'Score: 0.7', 'Score: 0.2']

#     evaluator = ContrastEvaluator()
#     evaluator.judge = DummyJudge(dummy_outputs)

#     scores = evaluator.evaluate(dataset)

#     expected_scores = [0.9 - 0.4, 0.7 - 0.2]
#     assert len(scores) == 2
#     for score, expected in zip(scores, expected_scores):
#         assert (
#             abs(score - expected) < 1e-6
#         ), 'Contrast scores should match the expected differences.'
