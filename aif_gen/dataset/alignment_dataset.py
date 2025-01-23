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
        return self._task

    @property
    def samples(self) -> List[AlignmentDatasetSample]:
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, key: Union[slice, int]
    ) -> Union[AlignmentDatasetSample, List[AlignmentDatasetSample]]:
        return self.samples[key]

    def append(self, sample: AlignmentDatasetSample) -> None:
        if isinstance(sample, AlignmentDatasetSample):
            self.samples.append(sample)
        else:
            raise TypeError(
                f'Sample: {sample} must be of type AlignmentDatasetSample but got {sample.__class__.__name__}'
            )

    def extend(self, samples: List[AlignmentDatasetSample]) -> None:
        for sample in samples:
            self.append(sample)

    def to_json(self, file_path: str) -> None:
        r"""Save the AlignmentDataset to a json file."""
        dataset_dict: Dict[str, Any] = {}
        dataset_dict['task'] = self.task.to_dict()
        dataset_dict['samples'] = []
        for sample in self.samples:
            dataset_dict['samples'].append(asdict(sample))

        with open(file_path, 'w') as f:
            json.dump(dataset_dict, f)

    @classmethod
    def from_json(cls, file_path: str) -> 'AlignmentDataset':
        r"""Load the AlignmentDataset to a json file."""
        with open(file_path, 'r') as f:
            dataset_dict = json.load(f)

        task = AlignmentTask.from_dict(dataset_dict['task'])
        samples = []
        for sample in dataset_dict['samples']:
            sample = AlignmentDatasetSample(**sample)
            samples.append(sample)

        return cls(task, samples)
