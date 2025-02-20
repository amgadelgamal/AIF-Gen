from typing import Dict, List

from sentence_transformers import SentenceTransformer, util

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset


class RelevanceEvaluator:
    def __init__(self) -> None:
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def evaluate(self, dataset: AlignmentDataset) -> Dict[str, int]:
        """For each sample in the AlignmentDataset, compute the cosine similarity between
        the prompt and the chosen response, and return a dictionary mapping sample identifiers
        to the computed relevance score (as an integer percentage).

        Args:
            dataset (AlignmentDataset): The dataset to evaluate.

        Returns:
            Dict[str, int]: A dictionary mapping sample IDs to computed relevance scores.
        """
        scores = {}
        for idx, sample in enumerate(dataset.samples):
            # Use sample.id if available; otherwise, use the index as the identifier.
            sample_id = str(getattr(sample, 'id', idx))
            prompt = sample.prompt
            chosen = sample.chosen
            embeddings = self.model.encode([prompt, chosen], convert_to_tensor=True)
            cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            score_int = int(round(cosine_sim * 100))
            scores[sample_id] = score_int
        return scores

    def relevance_evaluation(self, dataset: Dataset) -> List[Dict[str, int]]:
        """Compute the relevance score for each sample in the dataset.

        Args:
            dataset (Union[AlignmentDataset, ContinualAlignmentDataset]): The dataset to evaluate.

        Returns:
            List[Dict[str, int]]: For every AlignmentDataset, returns a dictionary mapping sample IDs
            to computed relevance scores (as integer percentages). If the input dataset is an AlignmentDataset,
            a one-element list is returned.
        """
        if isinstance(dataset, AlignmentDataset):
            datasets = [dataset]
        else:
            assert isinstance(dataset, ContinualAlignmentDataset)
            datasets = dataset.datasets

        results = []
        for ds in datasets:
            results.append(self.evaluate(ds))
        return results
