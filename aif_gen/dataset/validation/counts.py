from typing import Dict, List

from aif_gen.typing import Dataset


def count_validation(dataset: Dataset) -> List[Dict[str, int]]:
    r"""Count the number of 'unique' samples in the dataset.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.

    Returns:
        List[Dict[str, int]]: For every AligmentDataset, returns a dictionary with the following entries:

        'samples'           -> int: The total number of samples in the AlignmentDataset.
        'unique_samples'    -> int: The number of unique samples in the AlignmentDataset.
        'unique_prompts'    -> int: The number of unique prompts in the AlignmentDataset.
        'unique_chosen'     -> int: The number of unique chosen responses in the AlignmentDataset.
        'unique_rejected'   -> int: The number of unique rejected responses in the AlignmentDataset.

    Note:
        If the input dataset is an AlignmentDataset (non-continual), this function
        returns a 1 element list with the relevant statistics.
    """
    raise NotImplementedError()
