from typing import List

from aif_gen.dataset import AlignmentDataset
from aif_gen.dataset.validation.base import BaseMetric


class DiversityEvaluator(BaseMetric):
    """Computes the diversity for a set of responses via the inverse Self-BLEU score.
    A higher Self-BLEU score indicates lower diversity for generated sentences.

    References:
        - https://arxiv.org/pdf/1802.01886
        - https://github.com/geek-ai/Texygen

    Args:
        response_set (list[str]): A list of generated sentences.
        ngram (int): The maximum n-gram order for BLEU calculation. Default of 3 matches the original paper.
    """

    def __init__(self, ngram: int = 3):
        self._ensure_nltk_resources()
        self.ngram = ngram

    def compute_response_diversity(self, response_set: List[str]) -> float:
        # Avoid redundant calculations
        if not response_set or len(response_set) < 2:
            return 0.0

        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        # BLEU weight setting (e.g., for BLEU-3: (1/3, 1/3, 1/3))
        weight = tuple(1.0 / self.ngram for _ in range(self.ngram))

        # Tokenize responses for BLEU calculation
        tokenized_responses = [
            nltk.word_tokenize(sentence) for sentence in response_set
        ]

        scores = []
        for i, hypothesis in enumerate(tokenized_responses):
            other_responses = tokenized_responses[:i] + tokenized_responses[i + 1 :]
            score = sentence_bleu(
                other_responses,
                hypothesis,
                weight,
                smoothing_function=SmoothingFunction().method1,
            )
            scores.append(score)

        # Average self-BLEU score
        bleu_score = sum(scores) / len(scores)

        # Return the inverse BLEU score as diversity metric
        return 1.0 - bleu_score

    def evaluate(self, dataset: AlignmentDataset) -> List[float]:
        """Evaluate the diversity metric on an AlignmentDataset.

        Args:
            dataset (AlignmentDataset): The dataset to evaluate.

        Returns:
            List[float]: A list of diversity scores, one per sample.
        """
        response_set = [sample.chosen for sample in dataset.samples]
        overall_score = self.compute_response_diversity(response_set)

        return [overall_score] * len(dataset.samples)

    @staticmethod
    def _ensure_nltk_resources() -> None:
        import nltk

        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
