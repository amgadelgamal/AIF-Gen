import re
from typing import Dict, List

from transformers import pipeline

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset


class RelevanceEvaluator:
    def __init__(self) -> None:
        # Replace the similarity model with an LLM judge model.
        self.judge = pipeline('text-generation', model='gpt2', tokenizer='gpt2')

    def _parse_rating(self, text: str) -> float:
        """Extracts the first floating point number from the generated text.
        Returns a float between 0 and 1. If parsing fails, returns 0.5.
        """
        match = re.search(r'([0-9]*\.?[0-9]+)', text)
        if match:
            try:
                rating = float(match.group(1))
                return max(0.0, min(1.0, rating))
            except Exception:
                return 0.5
        return 0.5

    def evaluate(self, dataset: AlignmentDataset) -> Dict[str, int]:
        """For each sample in the AlignmentDataset, prompt the LLM to judge the relevance between
        the prompt and the chosen response. Returns a dictionary mapping sample identifiers
        to the computed relevance score (as an integer percentage).

        Args:
            dataset (AlignmentDataset): The dataset to evaluate.

        Returns:
            Dict[str, int]: A dictionary mapping sample IDs to computed relevance scores.
        """
        scores: Dict[str, int] = {}
        for idx, sample in enumerate(dataset.samples):
            # Use sample.id if available; otherwise, generate one from the index.
            sample_id: str = str(getattr(sample, 'id', idx))
            prompt = sample.prompt
            chosen = sample.chosen
            # Construct a prompt for the judge LLM.
            judge_prompt = (
                f'Please rate the relevance of the following response to the given prompt on a scale from 0 to 1, '
                f'where 1 means perfectly relevant.\n\n'
                f'Prompt: {prompt}\n'
                f'Response: {chosen}\n\n'
                f'Rating (0 to 1):'
            )
            # Generate a response from the LLM judge.
            output = self.judge(judge_prompt, max_length=50, do_sample=False)[0][
                'generated_text'
            ]
            rating = self._parse_rating(output)
            score_int = int(round(rating * 100))
            scores[sample_id] = score_int
        return scores

    def relevance_evaluation(self, dataset: Dataset) -> List[Dict[str, int]]:
        """Compute the relevance score for each sample in the dataset.

        Args:
            dataset (Union[AlignmentDataset, ContinualAlignmentDataset]): The dataset to evaluate.

        Returns:
            List[Dict[str, int]]: For each AlignmentDataset, returns a dictionary mapping sample IDs
            to computed relevance scores (as integer percentages). If the input is a single AlignmentDataset,
            a one-element list is returned.
        """
        if isinstance(dataset, AlignmentDataset):
            datasets = [dataset]
        else:
            # This assert is here to satisfy type checkers.
            assert isinstance(dataset, ContinualAlignmentDataset)
            datasets = dataset.datasets

        results = []
        for ds in datasets:
            results.append(self.evaluate(ds))
        return results
