from typing import Dict, List

from aif_gen.typing import Dataset


def entropy_validation(dataset: Dataset) -> List[Dict[str, float]]:
    r"""Report various entropy measures on tokens in the dataset samples.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.

    Returns:
        List[Dict[str, int]]: For every AligmentDataset, returns a dictionary with the following entries:

        'token_entropy'     -> float: The entropy across tokens (prompts and responses combined) for all samples in the AlignmentDataset.
        'prompt_entropy'    -> float: The entropy across prompts in samples of the AlignmentDataset.
        'chosen_entropy'    -> float: The entropy across chosen responses in samples of the AlignmentDataset.
        'rejected_entropy'  -> float: The entropy acorss rejected responses in the samples of the AlignmentDataset.

    Note:
        If the input dataset is an AlignmentDataset (non-continual), this function
        returns a 1 element list with the relevant statistics.
    """
    raise NotImplementedError()
