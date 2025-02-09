import json
from dataclasses import asdict
from typing import Any, Dict, List, Union

from aif_gen.task import AlignmentTask

from .alignment_sample import AlignmentDatasetSample


class AlignmentDataset:
    r"""Container object for an Alignment Dataset.

    Args:
        task (AligmnentTask): The AlignmentTask associated with the dataset.
        samples (List[AlignmentDatasetSample]): The samples in this AlignmentDataset.
    """

    def __init__(
        self, task: AlignmentTask, samples: List[AlignmentDatasetSample]
    ) -> None:
        self._task = task
        self._samples = samples

    @property
    def task(self) -> AlignmentTask:
        r"""AlignmentTask: The task associated with the AlignmentDataset."""
        return self._task

    @property
    def samples(self) -> List[AlignmentDatasetSample]:
        r"""List[AlignmentDatasetSample]: The list of samples associated with the AlignmentDataset."""
        return self._samples

    @property
    def num_samples(self) -> int:
        r"""int: The number of samples associated with the AlignmentDataset."""
        return len(self.samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, key: Union[slice, int]
    ) -> Union[AlignmentDatasetSample, List[AlignmentDatasetSample]]:
        # Slicing directly on the samples
        return self.samples[key]

    def to_json(self, file_path: str) -> None:
        r"""Save the AlignmentDataset to a json file.

        Note: Uses to_dict() under the hood to get a dictionary representation.

        Args:
            file_path (str): The os.pathlike object to write to.
        """
        dataset_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(dataset_dict, f)

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the AlignmentDataset to dictionary represenetation.

        Note: This method is the functional inverse of AlignmentDataset.from_dict().

        Returns:
            Dict[str, Any]: The dictionary representation of the AlignmentDataset.
        """
        dataset_dict: Dict[str, Any] = {}
        dataset_dict['task'] = self.task.to_dict()
        dataset_dict['samples'] = []
        for sample in self.samples:
            dataset_dict['samples'].append(asdict(sample))
        return dataset_dict

    @classmethod
    def from_json(cls, file_path: str) -> 'AlignmentDataset':
        r"""Load the AlignmentDataset from a json file.

        Note: Uses AlignmentDataset.from_dict() under the hood to parse the representation.

        Args:
            file_path (str): The os.pathlike object to read from.

        Returns:
            AlignmentDataset: The newly constructed AlignmentDataset.
        """
        with open(file_path, 'r') as f:
            dataset_dict = json.load(f)

        return cls.from_dict(dataset_dict)

    @classmethod
    def from_dict(cls, dataset_dict: Dict[str, Any]) -> 'AlignmentDataset':
        r"""Construct an AlignmentDataset from dictionary representation.

        Note:
            Expects 'task', and 'samples' keys to be present in the dictionary.
            The 'task' value should be parsable by AlignmentTask.from_dict().
            The 'samples' value should be a list of dictionaries, each of which
            are parsable by AlignmentDatasetSample.

        Args:
            dataset_dict (Dict[str, Any]): The dictionary that encodes the AlignmentDataset.

        Returns:
            AlignmentDataset: The newly constructed AlignmentDataset.

        Raises:
            ValueError: If the input dictionary is missing any required keys.
        """
        task = AlignmentTask.from_dict(dataset_dict['task'])
        samples = []
        for sample in dataset_dict['samples']:
            sample = AlignmentDatasetSample(**sample)
            samples.append(sample)

        return cls(task, samples)
