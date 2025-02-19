import json
import pathlib
from dataclasses import asdict
from typing import Any, Dict, List, Union

from datasets import Dataset

from aif_gen.task import AlignmentTask

from .alignment_sample import AlignmentDatasetSample


class AlignmentDataset:
    r"""Container object for an Alignment Dataset.

    Args:
        task (AligmnentTask): The AlignmentTask associated with the dataset.
        samples (List[AlignmentDatasetSample]): The samples in this AlignmentDataset.
        train_frac (float): Fraction of samples that belong to the training split.

    Raises:
        ValueError: If train_frac is not in the interval [0, 1.0]
    """

    def __init__(
        self,
        task: AlignmentTask,
        samples: List[AlignmentDatasetSample],
        train_frac: float = 1.0,
    ) -> None:
        self._task = task
        self._samples = samples

        if not (0 <= train_frac <= 1):
            raise ValueError(f'Train fraction must be in [0, 1] but got: {train_frac}')
        self._train_frac = train_frac

    @property
    def task(self) -> AlignmentTask:
        r"""AlignmentTask: The task associated with the AlignmentDataset."""
        return self._task

    @property
    def train_frac(self) -> float:
        r"""Fraction of samples that belong to the training split."""
        return self._train_frac

    @property
    def test_frac(self) -> float:
        r"""Fraction of samples that belong to the testing split."""
        return 1.0 - self._train_frac

    @property
    def samples(self) -> List[AlignmentDatasetSample]:
        r"""List[AlignmentDatasetSample]: The list of samples associated with the AlignmentDataset."""
        return self._samples

    @property
    def train(self) -> List[AlignmentDatasetSample]:
        r"""List[AlignmentDatasetSample]: The list of training samples associated with the AlignmentDataset."""
        return self.samples[: self.num_train_samples]

    @property
    def test(self) -> List[AlignmentDatasetSample]:
        r"""List[AlignmentDatasetSample]: The list of testing samples associated with the AlignmentDataset."""
        return self.samples[self.num_train_samples :]

    @property
    def num_samples(self) -> int:
        r"""int: The number of samples associated with the AlignmentDataset."""
        return len(self.samples)

    @property
    def num_train_samples(self) -> int:
        r"""int: The number of training samples associated with the AlignmentDataset."""
        return int(self.train_frac * len(self.samples))

    @property
    def num_test_samples(self) -> int:
        r"""int: The number of test samples associated with the AlignmentDataset."""
        return self.num_samples - self.num_train_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, key: Union[slice, int]
    ) -> Union[AlignmentDatasetSample, List[AlignmentDatasetSample]]:
        # Slicing directly on the samples
        return self.samples[key]

    def to_json(self, file_path: Union[str, pathlib.Path]) -> None:
        r"""Save the AlignmentDataset to a json file.

        Note: Uses to_dict() under the hood to get a dictionary representation.

        Args:
            file_path (Union[str, pathlib.Path]): The os.pathlike object to write to.
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
        dataset_dict['train'] = []
        dataset_dict['test'] = []

        for sample in self.train:
            dataset_dict['train'].append(asdict(sample))

        for sample in self.test:
            dataset_dict['test'].append(asdict(sample))

        return dataset_dict

    @classmethod
    def from_json(cls, file_path: Union[str, pathlib.Path]) -> 'AlignmentDataset':
        r"""Load the AlignmentDataset from a json file.

        Note: Uses AlignmentDataset.from_dict() under the hood to parse the representation.

        Args:
            file_path (Union[str, pathlib.Path]): The os.pathlike object to read from.

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
            Expects 'task', and 'train', 'test' keys to be present in the dictionary.
            The 'task' value should be parsable by AlignmentTask.from_dict().
            The 'train' and 'test' value should be a list of dictionaries, each of which
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
        for sample in dataset_dict['train']:
            sample = AlignmentDatasetSample(**sample)
            samples.append(sample)

        num_train_samples = len(samples)

        for sample in dataset_dict['test']:
            sample = AlignmentDatasetSample(**sample)
            samples.append(sample)

        train_frac = num_train_samples / len(samples)
        return cls(task, samples, train_frac)

    def to_hf_compatible(self) -> dict[str, Dataset]:
        r"""Convert the AlignmentDataset to a dictionary compatible with HuggingFace datasets.

        Returns:
            dict[str, Dataset]: The dictionary compatible with HuggingFace datasets.
        """
        dataset_dict: dict[str, Any] = self.to_dict()

        hf_dict: dict[str, Dataset] = {
            'train': Dataset.from_dict(
                {
                    'prompt': [sample['prompt'] for sample in dataset_dict['train']],
                    'chosen': [sample['chosen'] for sample in dataset_dict['train']],
                    'rejected': [
                        sample['rejected'] for sample in dataset_dict['train']
                    ],
                },
                split='train',
            ),
            'test': Dataset.from_dict(
                {
                    'prompt': [sample['prompt'] for sample in dataset_dict['test']],
                    'chosen': [sample['chosen'] for sample in dataset_dict['test']],
                    'rejected': [sample['rejected'] for sample in dataset_dict['test']],
                },
                split='test',
            ),
        }
        return hf_dict
