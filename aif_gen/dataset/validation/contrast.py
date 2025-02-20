import re
from typing import List

from transformers import pipeline

from aif_gen.dataset import AlignmentDataset
from aif_gen.dataset.validation.base import BaseMetric


class ContrastEvaluator(BaseMetric):
    """A contrast score evaluator that computes the difference between the LLM-generated scores
    for the chosen and rejected responses. This class inherits from BaseMetric and uses a
    text-generation pipeline as an LLM judge.
    """

    def __init__(self) -> None:
        self.judge = pipeline('text-generation', model='gpt2', tokenizer='gpt2')

    def _parse_rating(self, text: str) -> float:
        """Extracts the first floating point number from the generated text.
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
        """For each sample in the dataset, uses the LLM judge to evaluate both the chosen and
        rejected responses. The contrast score is computed as the difference between the
        chosen and rejected scores (each between 0 and 1) and is returned as a float.

        Args:
            dataset (AlignmentDataset): The dataset to evaluate.

        Returns:
            List[float]: A list of contrast scores (as floats) for each sample.
        """
        scores: List[float] = []
        for sample in dataset.samples:
            # Construct prompt for the chosen response.
            chosen_prompt = (
                'Please evaluate the following chosen response on a scale from 0 to 1, '
                'where 1 indicates excellent coherence and alignment, and 0 indicates poor quality:\n\n'
                f'Chosen Response: {sample.chosen}\n\n'
                'Score (0 to 1):'
            )

            chosen_output = self.judge(
                chosen_prompt,
                max_new_tokens=50,
                do_sample=False,
                truncation=True,
                pad_token_id=50256,
            )[0]['generated_text']
            chosen_score = self._parse_rating(chosen_output)

            # Construct prompt for the rejected response.
            rejected_prompt = (
                'Please evaluate the following rejected response on a scale from 0 to 1, '
                'where 1 indicates excellent coherence and alignment, and 0 indicates poor quality:\n\n'
                f'Rejected Response: {sample.rejected}\n\n'
                'Score (0 to 1):'
            )

            rejected_output = self.judge(
                rejected_prompt,
                max_new_tokens=50,
                do_sample=False,
                truncation=True,
                pad_token_id=50256,
            )[0]['generated_text']
            rejected_score = self._parse_rating(rejected_output)

            # Compute contrast as the difference between chosen and rejected scores.
            contrast = chosen_score - rejected_score
            scores.append(contrast)
        return scores
