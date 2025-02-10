from typing import List

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from datasets import load_dataset


def init_mock_dataset(dataset_name: str) -> ContinualAlignmentDataset:
    if dataset_name == "debug":
        datasets = get_debug_datasets()
    elif dataset_name == "ultrafeedback2anthropic":
        datasets = get_ultrafeedback2anthropic_datasets()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return ContinualAlignmentDataset(datasets)


def get_debug_datasets() -> List[AlignmentDataset]:
    datasets = [
        {
            "train": load_dataset("trl-lib/ultrafeedback_binarized", split="train").select(range(100)),
            "test": load_dataset("trl-lib/ultrafeedback_binarized", split="test").select(range(100)),
        },
        {
            "train": load_dataset("trl-lib/ultrafeedback_binarized", split="train").select(range(100, 200)),
            "test": load_dataset("trl-lib/ultrafeedback_binarized", split="test").select(range(100, 200)),
        },
        {
            "train": load_dataset("Anthropic/hh-rlhf", split="train").select(range(100)),
            "test": load_dataset("Anthropic/hh-rlhf", split="test").select(range(100)),
        },
    ]
    return datasets


def get_ultrafeedback2anthropic_datasets() -> List[AlignmentDataset]:
    datasets = [
        load_dataset("trl-lib/ultrafeedback_binarized"),
        load_dataset("Anthropic/hh-rlhf"),
    ]
    return datasets
