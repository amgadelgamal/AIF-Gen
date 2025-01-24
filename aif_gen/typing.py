from typing import Union

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset

# Typedef for convenience
Dataset = Union[ContinualAlignmentDataset, AlignmentDataset]
