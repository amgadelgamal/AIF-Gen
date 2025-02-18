from typing import Callable, Dict, List

import nltk
from transformers import pipeline

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class ContrastEvaluator:
    """Computes the difference between the alignment scores of the chosen and rejected responses.
    It uses the same alignment function (based on the task preference) for both.
    """

    def __init__(self) -> None:
        self.classifier = pipeline('sentiment-analysis')

    def get_alignment_mode(self, dataset: AlignmentDataset) -> Callable[[dict], float]:
        """Determines the alignment scoring function based on the dataset's task preference.

        Returns:
            A function that maps a classifier result to a score (between 0 and 1).
        """
        preference = (
            dataset.task.preference.lower()
            if hasattr(dataset.task, 'preference')
            else ''
        )
        if 'polarizing' in preference:
            return lambda result: abs(result['score'] - 0.5) * 2
        elif 'negative' in preference:
            return (
                lambda result: result['score']
                if result['label'] == 'NEGATIVE'
                else 1 - result['score']
            )
        elif 'positive' in preference:
            return (
                lambda result: result['score']
                if result['label'] == 'POSITIVE'
                else 1 - result['score']
            )
        else:
            return (
                lambda result: result['score']
                if result['label'] == 'POSITIVE'
                else 1 - result['score']
            )

    def evaluate(self, dataset: AlignmentDataset) -> Dict[str, int]:
        """For each sample in the dataset, compute the alignment scores for both the chosen and rejected responses,
        then compute their difference (contrast) and return a dictionary mapping sample identifiers to these scores.

        Args:
            dataset (AlignmentDataset): The dataset to evaluate.

        Returns:
            Dict[str, int]: A dictionary mapping sample identifiers to computed contrast scores (as integer percentages).
        """
        scores = {}
        mode_func = self.get_alignment_mode(dataset)
        for idx, sample in enumerate(dataset.samples):
            # Use sample.id if available; otherwise, use the index as the identifier.
            sample_id = str(getattr(sample, 'id', idx))
            result_chosen = self.classifier(sample.chosen)[0]
            result_rejected = self.classifier(sample.rejected)[0]
            score_chosen = mode_func(result_chosen)
            score_rejected = mode_func(result_rejected)
            contrast = score_chosen - score_rejected
            score_int = int(round(contrast * 100))
            scores[sample_id] = score_int
        return scores

    def contrast_evaluation(self, dataset: Dataset) -> List[Dict[str, int]]:
        """Compute the contrast score for each sample in the dataset.

        Args:
            dataset (Union[AlignmentDataset, ContinualAlignmentDataset]): The dataset to evaluate.

        Returns:
            List[Dict[str, int]]: For every AlignmentDataset, returns a dictionary mapping sample identifiers
            to computed contrast scores (as integer percentages). If the input dataset is an AlignmentDataset,
            a one-element list is returned.
        """
        if isinstance(dataset, AlignmentDataset):
            datasets = [dataset]
        else:
            # This assert is here to satisfy type checkers.
            assert isinstance(dataset, ContinualAlignmentDataset)
            datasets = dataset.datasets

        results = []
        for ds in datasets:
            results.append(self.evaluate(ds))
        return results
