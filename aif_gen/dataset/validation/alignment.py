import re
from typing import List

from transformers import pipeline

from aif_gen.dataset import AlignmentDataset
from aif_gen.dataset.validation.base import BaseMetric


class AlignmentEvaluator(BaseMetric):
    def __init__(self) -> None:
        # Initialize the LLM judge using a text-generation pipeline.
        self.judge = pipeline('text-generation', model='gpt2', tokenizer='gpt2')

    def _parse_rating(self, text: str) -> float:
        """Extract the first floating point number from the generated text.
        Returns a float between 0 and 1. If parsing fails, returns 0.5.
        """
        match = re.search(r'([0-9]*\.?[0-9]+)', text)
        if match:
            try:
                rating = float(match.group(1))
                return max(0.0, min(1.0, rating))
            except Exception:
                return 0.5
        return 0.5

    def evaluate(self, dataset: AlignmentDataset) -> List[float]:
        """Evaluates the dataset and returns a tuple of:
        - A list of computed alignment scores (as floats) for each sample.
        - The fraction of samples where parsing the rating failed.
        """
        scores: List[float] = []
        failures = 0

        for sample in dataset.samples:
            prompt = sample.prompt
            chosen = sample.chosen
            rejected = sample.rejected

            judge_prompt = (
                'Please evaluate the alignment of the following responses with respect to the given prompt. '
                'Consider which response better aligns with the prompt in terms of clarity, relevance, and correctness. '
                'Provide a rating from 0 to 1 for the chosen response, where 1 indicates perfect alignment and 0 indicates no alignment at all, '
                'relative to the rejected response.\n\n'
                f'Prompt: {prompt}\n\n'
                f'Chosen Response: {chosen}\n\n'
                f'Rejected Response: {rejected}\n\n'
                'Alignment Score (0 to 1):'
            )

            output = self.judge(judge_prompt, max_length=50, do_sample=False)[0][
                'generated_text'
            ]
            rating = self._parse_rating(output)

            # Check if the rating is the fallback value.
            if rating == 0.5 and '0.5' not in output:
                failures += 1

            scores.append(rating)

        self.failure_rate = failures / len(dataset.samples) if dataset.samples else 0.0
        return scores
