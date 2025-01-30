from typing import Dict, List, Optional, Tuple

from aif_gen.task import AlignmentTask, DomainComponent

from .base import PromptMapperBase


class PromptMapper(PromptMapperBase):
    r"""Generate a prompt that, when given to a language model, produces a prompt for a given AlignmentTask.

    Samples domain component seed words from the AlignmentTask to contextualize the prompt.
    The sampling is parameterized by the weight of each component of the domain.

    Args:
        min_words_per_domain_component (int): Minimum number of seed words to add to the prompt for each domain component (default=1)
        max_words_per_domain_component (int): Maximum number of seed words to add to the prompt for each domain component (default=10)
        suffix_context (Optional[str]=None): Optionally added suffix context into the generated prompt.
    """

    def __init__(
        self,
        min_words_per_domain_component: int = 1,
        max_words_per_domain_component: int = 10,
        suffix_context: Optional[str] = None,
    ) -> None:
        if min_words_per_domain_component < 0:
            raise ValueError(
                f'Min words per domain component must be non-negative, got: {min_words_per_domain_component}'
            )
        if max_words_per_domain_component < min_words_per_domain_component:
            raise ValueError(
                f'Max words per domain compoment ({max_words_per_domain_component}) '
                f'must be >= Min words per domain component ({min_words_per_domain_component})'
            )

        self._min_words_per_domain_component = min_words_per_domain_component
        self._max_words_per_domain_component = max_words_per_domain_component
        self._suffix_context = suffix_context

    def generate_prompt(self, task: AlignmentTask) -> str:
        # Normalize the domain component weights
        normalized_components = self._normalize_domain_weights(
            task.domain._components, task.domain._weights
        )

        # Sample seed words from each domain component
        sampled_words = self._sample_seed_words(normalized_components)

        # Generate the final prompt, including preferences, ethical guidelines, and seed word rules
        return self._construct_prompt(task.objective, sampled_words, task.preference)

    @property
    def min_words_per_domain_component(self) -> int:
        r"""Minimum number of seed words to add to the prompt for each domain component."""
        return self._min_words_per_domain_component

    @property
    def max_words_per_domain_component(self) -> int:
        r"""Maximum number of seed words to add to the prompt for each domain component."""
        return self._max_words_per_domain_component

    @property
    def suffix_context(self) -> Optional[str]:
        f"""Optional added suffix context into the generated prompt."""
        return self._suffix_context

    def _normalize_domain_weights(
        self, components: List[DomainComponent], weights: List[float]
    ) -> Dict[str, Tuple]:
        total_weight = sum(weights)
        return {
            component.name: (component, weight / total_weight)
            for component, weight in zip(components, weights)
        }

    def _sample_seed_words(self, components: dict) -> List[str]:
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
        seed_words_str = ', '.join(seed_words)

        # Define ethical guidelines as an additional part of the prompt
        ethical_guidelines = "Ensure that the generated response adheres to ethical practices, avoids biases, and respects the target audience's needs."

        return (
            f'Generate a prompt for an RLHF task. Use the following words in your prompt: {seed_words_str}.\n'
            f'The goal of the task is: {objective}.\n'
            f'Preference: {preference}.\n'
            f'{ethical_guidelines}'
        )
