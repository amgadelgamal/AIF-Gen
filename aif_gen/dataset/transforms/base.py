from abc import ABC, abstractmethod
from typing import Any, Union

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset

# Typedef for convenience
Dataset = Union[ContinualAlignmentDataset, AlignmentDataset]


class DatasetTransform(ABC):
    r"""Base class for transforming Alignment Datasets."""

    @abstractmethod
    def apply(self, dataset: Dataset, *args: Any, **kwargs: Any) -> Dataset:
        r"""Apply the transform onto a dataset.

        Args:
            dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to transform.
            args (Any): Optional positional arguments.
            kwargs (Any): Optional keyword arguments.

        Returns:
            Union[ContinualAlignmentDataset, AlignmentDataset]: The transformed dataset.
        """

    def __call__(self, dataset: Dataset, *args: Any, **kwargs: Any) -> Dataset:
        return self.apply(dataset, args, kwargs)

    def __str__(self) -> str:
        r"""Returns summary properties of the dynamic graph."""
        return self.__class__.__name__

    def _is_dataset_continual(self, dataset: Dataset) -> bool:
        return isinstance(dataset, ContinualAlignmentDataset)
