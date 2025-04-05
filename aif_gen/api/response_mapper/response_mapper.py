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

    @property
    def suffix_context(self) -> Optional[str]:
        f"""Optional added suffix context into the generated prompt."""
        return self._suffix_context
