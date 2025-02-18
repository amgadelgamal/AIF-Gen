from typing import Dict, List

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from multiprocessing import Pool

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def calc_bleu(response, hypothesis, weight) -> float:
    """Computes the BLEU score for a single response against others.
    Needs to be declared outside of class for parallel processing.
    """
    return sentence_bleu(
        response, hypothesis, weight, smoothing_function=SmoothingFunction().method1
    )


class DiversityEvaluator:
    """Computes the diversity for a set of responses via the inverse Self-BLEU score.
    A higher Self-BLEU score indicates lower diversity for generated sentences,
    therefore we return the inverse score.
    https://arxiv.org/pdf/1802.01886
    https://github.com/geek-ai/Texygen

    Args:
        response_set (list[str]): A list of generated sentences.
        ngram (int): The maximum n-gram order for BLEU calculation. Default of 3 matches the original paper.
        parallel (bool): Whether to compute BLEU scores in using Pool multiprocessing.
    """
    def __init__(self, ngram: int = 1, parallel: bool = False):
        self.ngram = ngram
        self.parallel = parallel
        
    def compute_response_diversity(self, response_set: list[str]) -> float:
        # Avoid redundant calculations
        if not response_set or len(response_set) < 2:
            return 0.0

        # BLEU weight setting (e.g., for BLEU-3: (1/3, 1/3, 1/3))
        weight = tuple(1.0 / self.ngram for _ in range(self.ngram))

        # Tokenize responses for BLEU calculation
        tokenized_responses = [
            nltk.word_tokenize(sentence) for sentence in response_set
        ]

        scores = []

        if self.parallel:
            # Compute BLEU scores in parallel
            with Pool() as pool:
                scores = pool.starmap(
                    calc_bleu,
                    [
                        (
                            tokenized_responses[:i] + tokenized_responses[i + 1 :],
                            hypothesis,
                            weight,
                        )
                        for i, hypothesis in enumerate(tokenized_responses)
                    ],
                )
        else:
            # Compute BLEU scores sequentially
            for i, hypothesis in enumerate(tokenized_responses):
                other_responses = tokenized_responses[:i] + tokenized_responses[i + 1 :]
                score = calc_bleu(other_responses, hypothesis, weight)
                scores.append(score)

        # Average self-BLEU score
        bleu_score = sum(scores) / len(scores)

        # Return the inverse BLEU score as diversity metric
        return 1.0 - bleu_score
    
    def evaluate(self, dataset: Dataset) -> List[Dict[str, int]]:
        """For a given AlignmentDataset (single task), compute the diversity between
        the chosen responses.

        Returns:
            List[Dict[str, int]]: A dictionary with a key corresponding to the task.
        """
        if isinstance(dataset, AlignmentDataset):
            datasets = [dataset]
        else:
            # This assert is here to make mypy happy
            assert isinstance(dataset, ContinualAlignmentDataset)
            datasets = dataset.datasets

        results = []
        for dataset in datasets:
            response_set = [sample.chosen for sample in dataset.samples]
            score = self.compute_response_diversity(response_set)
            results.append({str(dataset.task) : score})
        return results
