import logging
import re
from typing import Dict, List

from transformers import pipeline

from aif_gen.dataset import AlignmentDataset
from aif_gen.dataset.validation.base import BaseMetric


class AlignmentEvaluator(BaseMetric):
    def __init__(self) -> None:
        self.judge = pipeline('text-generation', model='gpt2', tokenizer='gpt2')
        self._failures = 0

    def evaluate(self, dataset: AlignmentDataset) -> Dict[List[float]]:
        scores = {
            'chosen_alignment': [],
            'rejected_alignment': [],
            'alignment_contrast': [],
        }

        for sample in dataset.samples:
            chosen_score = self._get_score(sample.prompt, sample.chosen)
            rejected_score = self._get_sore(sample.prompt, sample.rejected)
            scores['chosen_alignment'].append(chosen_score)
            scores['rejected_alignment'].append(rejected_score)
            scores['alignment_contrast'].append(chosen_score - rejected_score)

        if self._failures > 0:
            logging.warning(
                f'{self.__class__.__name__} evaluation parsing failures: {self._failures} / {2 * dataset.samples}'
            )
        return scores

    def _get_score(self, prompt: str, response: str) -> float:
        judge_prompt = (
            'Please evaluate the alignment of the following response with respect to the given prompt. '
            'Consider whether the response aligns with the prompt in terms of clarity, relevance, and correctness. '
            'Provide a rating from 0 to 1 where 1 indicates perfect alignment and 0 indicates no alignment at all.\n\n'
            f'Prompt: {prompt}\n\n'
            f'Response: {response}\n\n'
            'Alignment Score (0 to 1):'
        )
        output = self.judge(
            judge_prompt,
            max_new_tokens=50,
            do_sample=False,
            truncation=True,
            pad_token_id=50256,
        )[0]['generated_text']
        return self._parse_rating(output)

    def _parse_rating(self, text: str) -> float:
        match = re.search(r'([-+]?[0-9]*\.?[0-9]+)', text)
        try:
            rating = float(match.group(1))
            return max(0.0, min(1.0, rating))
        except Exception:
            self._failures += 1
            return 0.5
