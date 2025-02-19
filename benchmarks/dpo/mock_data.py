from pathlib import Path
from typing import Any, Union

from datasets import Dataset, load_dataset

from aif_gen.dataset import ContinualAlignmentDataset


def init_mock_dataset(dataset_name: str) -> list[dict[str, Dataset]]:
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
) -> list[dict[str, Dataset]]:
    # if isinstance(dataset, ContinualAlignmentDataset):
    #     return dataset.
    raise NotImplementedError


def get_debug_datasets() -> list[dict[str, Any]]:
    datasets = [
        {
            'train': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='train'
            ).select(range(100)),
            'test': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='test'
            ).select(range(100)),
        },
        {
            'train': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='train'
            ).select(range(100, 200)),
            'test': load_dataset(
                'trl-lib/ultrafeedback_binarized', split='test'
            ).select(range(100, 200)),
        },
        {
            'train': load_dataset('Anthropic/hh-rlhf', split='train').select(
                range(100)
            ),
            'test': load_dataset('Anthropic/hh-rlhf', split='test').select(range(100)),
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


if __name__ == '__main__':
    ca = init_mock_dataset('debug')
