import json
from typing import Any, Dict, List, Union

from .alignment_dataset import AlignmentDataset


class ContinualAlignmentDataset:
    r"""Container object for a Continual Alignment Dataset.

    Args:
        datasets (List[ContinualAlignmentDataset]): Temporal list of AlignmentDatasets constituents.
    """

    def __init__(self, datasets: List[AlignmentDataset]) -> None:
        self._datasets = datasets

    @property
    def datasets(self) -> List[AlignmentDataset]:
        return self._datasets

    @property
    def num_datasets(self) -> int:
        return len(self.datasets)

    @property
    def num_samples(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, key: Union[slice, int]
    ) -> Union[AlignmentDataset, List[AlignmentDataset]]:
        return self.datasets[key]

    def append(self, dataset: AlignmentDataset) -> None:
        if isinstance(dataset, AlignmentDataset):
            self.datasets.append(dataset)
        else:
            raise TypeError(
                f'Dataset: {dataset} must be of type AlignmentDataset but got {dataset.__class__.__name__}'
            )

    def extend(self, datasets: List[AlignmentDataset]) -> None:
        for dataset in datasets:
            self.append(dataset)

    def to_json(self, file_path: str) -> None:
        r"""Save the ContinualAlignmentDataset to a json file."""
        dataset_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(dataset_dict, f)

    def to_dict(self) -> Dict[str, Any]:
        dataset_dict: Dict[str, List[Any]] = {'datasets': []}
        for dataset in self.datasets:
            dataset_dict['datasets'].append(dataset.to_dict())
        return dataset_dict

    @classmethod
    def from_json(cls, file_path: str) -> 'ContinualAlignmentDataset':
        r"""Load the AlignmentDataset to a json file."""
        with open(file_path, 'r') as f:
            dataset_dict = json.load(f)
        return cls.from_dict(dataset_dict)

    @classmethod
    def from_dict(cls, dataset_dict: Dict[str, Any]) -> 'ContinualAlignmentDataset':
        datasets = []
        for dataset in dataset_dict['datasets']:
            datasets.append(AlignmentDataset.from_dict(dataset))
        return cls(datasets)
