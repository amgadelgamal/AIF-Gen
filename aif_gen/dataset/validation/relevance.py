import re
from typing import List

from transformers import pipeline

from aif_gen.dataset import AlignmentDataset
from aif_gen.dataset.validation.base import BaseMetric


class RelevanceEvaluator(BaseMetric):
    """A relevance evaluator that uses an LLM judge to assess how relevant a response is
    to its associated prompt. For each sample, the judge model is prompted to generate a
    relevance score between 0 and 1, where 1 indicates high relevance and 0 indicates low relevance.
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
        """For each sample in the dataset, uses the LLM judge to evaluate the relevance of the response
        with respect to the provided prompt. The relevance score is a float between 0 and 1.

        Args:
            dataset (AlignmentDataset): The dataset to evaluate.

        Returns:
            List[float]: A list of relevance scores for each sample.
        """
        scores: List[float] = []
        for sample in dataset.samples:
            # Construct a prompt for the judge to score the relevance.
            judge_prompt = (
                'Please evaluate the relevance of the following response with respect to the provided prompt. '
                'Rate the relevance on a scale from 0 to 1, where 1 indicates that the response is highly relevant, '
                'and 0 indicates that it is not relevant at all.\n\n'
                f'Prompt: {sample.prompt}\n\n'
                f'Response: {sample.chosen}\n\n'
                'Score (0 to 1):'
            )
            output = self.judge(
                judge_prompt,
                max_new_tokens=50,
                do_sample=False,
                truncation=True,
                pad_token_id=50256,
            )[0]['generated_text']

            relevance_score = self._parse_rating(output)
            scores.append(relevance_score)
        return scores
