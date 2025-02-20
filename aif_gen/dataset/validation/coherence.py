import re
from typing import List

from transformers import pipeline

from aif_gen.dataset import AlignmentDataset
from aif_gen.dataset.validation.base import BaseMetric


def _ensure_nltk_resources() -> None:
    import nltk

    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


class CoherenceEvaluator(BaseMetric):
    """A coherence evaluator that computes the coherence score for the chosen response of each sample
    using an LLM judge. This class lazily downloads NLTK resources and returns a list of floats,
    one per sample.
    """

    def __init__(self) -> None:
        _ensure_nltk_resources()
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
        """For each sample in the dataset, uses the LLM judge to evaluate the coherence of the chosen response.
        The coherence score is extracted from the LLM output and returned as a float.

        Args:
            dataset (AlignmentDataset): The dataset to evaluate.

        Returns:
            List[float]: A list of coherence scores (as floats) for each sample.
        """
        scores: List[float] = []
        for sample in dataset.samples:
            prompt = (
                'Please evaluate the coherence of the following response on a scale from 0 to 1, '
                'where 1 indicates excellent coherence and 0 indicates poor coherence:\n\n'
                f'Response: {sample.chosen}\n\n'
                'Coherence Score (0 to 1):'
            )
            output = self.judge(prompt, max_length=50, do_sample=False)[0][
                'generated_text'
            ]
            score = self._parse_rating(output)
            scores.append(score)
        return scores
