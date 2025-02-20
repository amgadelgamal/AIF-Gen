from abc import ABC, abstractmethod
from typing import List

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset


class BaseMetric(ABC):
    @abstractmethod
    def evaluate(self, dataset: AlignmentDataset) -> List[float]:
        """Evaluate the metric on an AlignmentDataset.

        Args:
            dataset (AlignmentDataset): The dataset to evaluate.

        Returns:
            List[int]: A list of scores for each sample in the dataset.
        """

    def evaluate_all(self, dataset: Dataset) -> List[List[float]]:
        """Evaluate the metric across multiple datasets.

        Args:
            dataset (Union[AlignmentDataset, ContinualAlignmentDataset]): The dataset(s) to evaluate.

        Returns:
            List[List[int]]: A list containing a list of scores for each dataset.
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
