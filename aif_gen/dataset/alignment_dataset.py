from typing import Union

from aif_gen.task import AlignmentTask

from .alignment_sample import AlignmentDatasetSampleBase


class AlignmentDataset:
    r"""Container object for an Alignment Dataset.

    Args:
        task (AligmnentTask): The AlignmentTask associated with the dataset.
    """

    def __init__(self, task: AlignmentTask) -> None:
        self._task = task

    @property
    def task(self) -> AlignmentTask:
        return self._task

    def __getitem__(self, key: Union[slice, int]) -> AlignmentDatasetSampleBase:
        raise NotImplementedError()

    def to_csv(self, file_path: str) -> None:
        r"""Save the AlignmentDataset to a file path."""
        raise NotImplementedError()

    @classmethod
    def from_csv(cls, file_path: str) -> 'AlignmentDataset':
        r"""Load the AlignmentDataset from a file path."""
        raise NotImplementedError()
