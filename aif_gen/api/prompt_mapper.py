from typing import List, Dict
from aif_gen.dataset.continual_alignment_dataset import ContinualAlignmentDataset
from aif_gen.task import AlignmentTask



class PromptMapper:
    def __init__(self):
        """
        Initialize the PromptMapper.
        """
        pass

    def generate_prompts(self, continual_dataset: ContinualAlignmentDataset) -> List[str]:
        """
        Generate prompts from a ContinualAlignmentDataset.

        Args:
            continual_dataset (ContinualAlignmentDataset): The input dataset containing alignment datasets.

        Returns:
            List[str]: A list of prompt strings for LLMs.
        """
        prompts = []

        # Iterate over each alignment dataset in the continual dataset
        for alignment_dataset in continual_dataset.datasets:
            task: AlignmentTask = alignment_dataset.task  # The task associated with this dataset
            samples = alignment_dataset.samples  # The samples in this dataset

            # Generate prompts for each sample
            for sample in samples:
                prompt_text = self._generate_prompt_text(
                    task.objective, sample.prompt, sample.chosen, sample.rejected
                )
                prompts.append(prompt_text)

        return prompts

    def _generate_prompt_text(self, objective: str, prompt: str, chosen: str, rejected: str) -> str:
        """
        Generate a single prompt text combining the objective, prompt, and sample completions.

        Args:
            objective (str): The objective of the alignment task.
            prompt (str): The user-provided prompt.
            chosen (str): The preferred completion.
            rejected (str): The less preferred completion.

        Returns:
            str: A structured prompt string for the LM.
        """
        return (
            f"Objective: {objective}\n"
            f"Prompt: {prompt}\n"
            f"Option 1: {chosen}\n"
            f"Option 2: {rejected}\n"
            f"Which option better aligns with the objective and why?"
        )