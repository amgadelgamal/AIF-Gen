from pathlib import Path
from typing import Any, Union

from datasets import Dataset, load_dataset

from aif_gen.dataset import ContinualAlignmentDataset


def _init_mock_dataset(
    dataset_name: Union[str, ContinualAlignmentDataset, Path],
) -> list[dict[str, Dataset]]:
    if dataset_name == 'debug':
        datasets: list[dict[str, Any]] = get_debug_datasets()
    elif dataset_name == 'ultrafeedback2anthropic':
        datasets = get_ultrafeedback2anthropic_datasets()
    elif dataset_name == 'ultrafeedback2anthropic_reduced':
        datasets = get_ultrafeedback2anthropic_datasets_reduced()
    else:
        raise ValueError(f'Unknown mock dataset: {dataset_name}')
    return datasets


def init_continual_dataset(
    dataset: Union[str, ContinualAlignmentDataset, Path],
    mock: bool = True,
) -> list[dict[str, Dataset]]:
    r"""Initialize a continual dataset from a given dataset name or path or a ContinualAlignmentDataset Object."""
    if mock:
        return _init_mock_dataset(dataset)
    if isinstance(dataset, str):
        path: Path = Path(dataset)
        data: ContinualAlignmentDataset = ContinualAlignmentDataset.from_json(path)
    elif isinstance(dataset, Path):
        data = ContinualAlignmentDataset.from_json(dataset)
    return data.to_hf_compatible()


def get_debug_datasets() -> list[dict[str, Any]]:
    datasets = [
        {
            'train': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='train'
            ).select(range(100)),
            'test': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='test'
            ).select(range(100)),
            'descriptiveness': load_dataset(
                'trl-lib/ultrafeedback-prompt', split='train'
            ).select(range(200)),
        },
        {
            'train': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='train'
            ).select(range(100, 200)),
            'test': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='test'
            ).select(range(100, 200)),
            'descriptiveness': load_dataset(
                'trl-lib/ultrafeedback-prompt', split='train'
            ).select(range(200, 400)),
        },
        {
            'train': load_dataset('Anthropic/hh-rlhf', split='train').select(
                range(100)
            ),
            'test': load_dataset('Anthropic/hh-rlhf', split='test').select(range(100)),
            'descriptiveness': load_dataset(
                'trl-lib/ultrafeedback-prompt', split='train'
            ).select(range(400, 600)),
        },
    ]
    return datasets


def get_ultrafeedback2anthropic_datasets_reduced() -> list[dict[str, Any]]:
    datasets = [
        {
            'train': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='train'
            ).select(range(35200)),
            'test': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='test'
            ).select(range(1000)),
        },
        {
            'train': load_dataset('Anthropic/hh-rlhf', split='train').select(
                range(35200)
            ),
            'test': load_dataset('Anthropic/hh-rlhf', split='test').select(range(1000)),
        },
    ]
    return datasets


def get_ultrafeedback2anthropic_datasets() -> list[dict[str, Any]]:
    datasets = [
        load_dataset('trl-lib/ultrafeedback_binarized'),
        load_dataset('Anthropic/hh-rlhf'),
    ]
    return datasets
