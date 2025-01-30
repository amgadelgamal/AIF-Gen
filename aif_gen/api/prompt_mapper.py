from typing import Dict, List, Tuple

from aif_gen.task import AlignmentTask, DomainComponent


class PromptMapper:
    def __init__(self) -> None:
        """Initialize the PromptMapper."""

    def generate_prompt(self, task: AlignmentTask) -> str:
        """Generate a single prompt based on the AlignmentTask, including task preferences, ethical guidelines,
        and seed word usage rules.

        Args:
            task (AlignmentTask): The alignment task containing the domain, objective, and preferences.

        Returns:
            str: A structured prompt string for the LLM.
        """
        # Normalize the domain component weights
        normalized_components = self._normalize_domain_weights(
            task.domain._components, task.domain._weights
        )

        # Sample seed words from each domain component
        sampled_words = self._sample_seed_words(normalized_components)

        # Generate the final prompt, including preferences, ethical guidelines, and seed word rules
        return self._construct_prompt(task.objective, sampled_words, task.preference)

    def _normalize_domain_weights(
        self, components: List[DomainComponent], weights: List[float]
    ) -> Dict[str, Tuple]:
        """Normalize the weights of the domain components.

        Args:
            components (dict): A dictionary of domain components with their weights.
            weights (List[float]): A list of corresponding weights for the components.

        Returns:
            dict: A dictionary with normalized weights and a tuple of component and weight.
        """
        total_weight = sum(weights)
        return {
            component.name: (component, weight / total_weight)
            for component, weight in zip(components, weights)
        }

    def _sample_seed_words(self, components: dict) -> List[str]:
        """Sample seed words from each domain component based on normalized weights.

        Args:
            components (dict): A dictionary of domain components with normalized weights.

        Returns:
            List[str]: A list of sampled seed words.
        """
        sampled_words = []
        for _, (component, normalized_weight) in components.items():
            # Assume each domain component has an attribute `seed_words`
            seed_words = component.seed_words
            num_words_to_sample = max(1, int(len(seed_words) * normalized_weight))
            sampled_words.extend(seed_words[:num_words_to_sample])

        return sampled_words

    def _construct_prompt(
        self, objective: str, seed_words: List[str], preference: str
    ) -> str:
        """Construct the final prompt string, including preferences, ethical guidelines, and seed word rules.

        Args:
            objective (str): The objective of the alignment task.
            seed_words (List[str]): The refined list of seed words.
            preference (str): The preference of the alignment task.

        Returns:
            str: The final prompt string.
        """
        seed_words_str = ', '.join(seed_words)

        # Define ethical guidelines as an additional part of the prompt
        ethical_guidelines = "Ensure that the generated response adheres to ethical practices, avoids biases, and respects the target audience's needs."

        return (
            f'Generate a prompt for an RLHF task. Use the following words in your prompt: {seed_words_str}.\n'
            f'The goal of the task is: {objective}.\n'
            f'Preference: {preference}.\n'
            f'{ethical_guidelines}'
        )
