from textwrap import dedent
from typing import Optional

from aif_gen.task import AlignmentTask

from .base import ResponseMapperBase


class ResponseMapper(ResponseMapperBase):
    r"""Generate a prompt that, when given to a language model, produces a winning and losing response to the task_prompt.

    Args:
        suffix_context (Optional[str]=None): Optionally add arbitrary context at the end of the generated prompt.
    """

    def __init__(self, suffix_context: Optional[str] = None) -> None:
        self._suffix_context = suffix_context
        self._preference_axes = [
            ('short', 'long'),
            ('formal', 'causal'),
            ('helpful', 'harmful'),
            ('angry', 'sad'),
            ('expert', 'eli5'),
            ('direct', 'hinted'),
            ('authoritative', 'tentative'),
            ('friendly', 'distance'),
            ('optimistic', 'pessimistic'),
            ('serious', 'humorous'),
            ('respectful', 'disrespectful'),
            ('complex', 'simple'),
            ('supportive', 'challenging'),
            ('neutral', 'biased'),
            ('detailed', 'abstract'),
            ('urgent', 'relaxed'),
            ('technical', 'layperson-friendly'),
        ]  # TODO could be added to the config

    def generate_prompt(self, task: AlignmentTask, task_prompt: str) -> str:
        prompt = f"""\
        Generate a 'chosen' and 'rejected' response pair to the following prompt: '{task_prompt}'.\n

        The 'chosen' response should better respond to the prompt according to the following preference: '{task.preference}'.
        The 'rejected' response should still respond to the prompt in a meaningful way, but should be worse or do not abide by (according to) the following preference: '{task.preference}'.

        You don't need to start your prompt by saying 'User asks' or start by "chosen:" or "rejected:".
        {self.ETHICAL_GUIDELINES}
        """
        if self.suffix_context:
            prompt += self.suffix_context

        prompt = dedent(prompt)
        return prompt

    def generate_no_preference_prompt(
        self, task: AlignmentTask, task_prompt: str, scores: list[int]
    ) -> str:
        """Generate prompt without explicitly stating the user preference."""
        prompt = f"""\
        Generate a response to the following prompt: '{task_prompt}'.\n

        {self.preference_axes_scale(scores)}
        You don't need to start your response by saying "here is the response" nor to give any meta-explanation. Just provide the response.
        {self.ETHICAL_GUIDELINES}
        """
        if self.suffix_context:
            prompt += self.suffix_context

        prompt = dedent(prompt)
        return prompt

    @property
    def preference_axes(self) -> list[tuple[str, str]]:
        return self._preference_axes

    @property
    def suffix_context(self) -> Optional[str]:
        f"""Optional added suffix context into the generated prompt."""
        return self._suffix_context

    def preference_axes_scale(
        self, scores: list[int], min_score: int = 1, max_score: int = 5
    ) -> str:
        """Generate a guide string for the preference axes scale to paste into the response meta-prompt.

        Args:
            scores (list[int]): List of scores for the preference axes.
            min_score (int, optional): Minimum score for the preference axes scale. Defaults to 1.
            max_score (int, optional): Maximum score for the preference axes scale. Defaults to 5.

        Returns:
            str: Description of the preference axes scale.
        """
        description: str = ''
        for i, axis in enumerate(self.preference_axes):
            description += f'On a scale of {min_score} to {max_score} where {min_score} is {axis[0]} and {max_score} is {axis[1]}, your response should be: {scores[i]} \n'
        description += 'Please ensure your responses align with the provided scores.'
        return description
