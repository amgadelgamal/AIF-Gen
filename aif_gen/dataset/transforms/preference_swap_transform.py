from .base import Dataset, DatasetTransform


class PreferenceSwapTransform(DatasetTransform):
    def apply(self, dataset: Dataset) -> Dataset:
        r"""Apply the transform onto a dataset.

        Args:
            dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to transform.
            args (Any): Optional positional arguments.
            kwargs (Any): Optional keyword arguments.

        Returns:
            Union[ContinualAlignmentDataset, AlignmentDataset]: The transformed dataset.
        """
        return dataset
