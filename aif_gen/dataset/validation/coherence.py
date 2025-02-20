from typing import Callable, Dict, List

from transformers import pipeline

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset


class CoherenceEvaluator:
    def __init__(self) -> None:
        self.classifier = pipeline('sentiment-analysis')

    def get_alignment_mode(self, dataset: AlignmentDataset) -> Callable[[dict], float]:
        """Determines the alignment scoring function based on the task's preference.

        Args:
            dataset (AlignmentDataset): The dataset whose task preference is used.

        Returns:
            Callable[[dict], float]: A function that maps a classifier result to a score (between 0 and 1).
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
        """For each sample in the dataset, run the sentiment classifier on the chosen response
        and compute the alignment score using the determined alignment mode.

        Returns a dictionary that maps each sample's identifier to its alignment score (as an integer percentage).

        Args:
            dataset (AlignmentDataset): The dataset to evaluate.

        Returns:
            Dict[str, int]: A dictionary mapping sample identifiers to computed alignment scores.
        """
        scores = {}
        mode_func = self.get_alignment_mode(dataset)
        for idx, sample in enumerate(dataset.samples):
            # Use sample.id if available; otherwise, use the index (as a string) as the sample identifier.
            sample_id = str(getattr(sample, 'id', idx))
            chosen = sample.chosen
            result = self.classifier(chosen)[0]
            score = mode_func(result)
            score_int = int(round(score * 100))
            scores[sample_id] = score_int
        return scores

    def alignment_evaluation(self, dataset: Dataset) -> List[Dict[str, int]]:
        """Compute the alignment score for each sample in the dataset.

        Args:
            dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to evaluate.

        Returns:
            List[Dict[str, int]]: For each AlignmentDataset, returns a dictionary mapping sample identifiers
            to computed alignment scores (as integer percentages). If the input dataset is an AlignmentDataset,
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
