from typing import List, Optional


class DomainComponent:
    r"""A domain component is an alias for a set of 'seed words' that describe a
    specific sphere of activity or knowledge.

    Args:
        name (str): The name that describes the content of the domain component.
        seed_words (List[str]): The list of seed words that describe the domain component.
        description (Optional[str]): An optional description of the domain component.
    """

    def __init__(
        self,
        name: str,
        seed_words: List[str],
        description: Optional[str] = None,
    ) -> None:
        if not len(seed_words):
            raise ValueError(
                'Cannot initialize a DomainComponent with an empty list of seed words'
            )

        self._name = name
        self._seed_words = seed_words
        self._description = description

    def __str__(self) -> str:
        s = f'{self._name} '
        if self.description is not None:
            s += f'({self.description}) '

        if len(self.seed_words) > 3:
            s += str(self._seed_words[:3])[:-1] + ', ...]'
        else:
            s += str(self._seed_words)

        return s

    @property
    def name(self) -> str:
        return self._name

    @property
    def seed_words(self) -> List[str]:
        return self._seed_words

    @property
    def num_seed_words(self) -> int:
        return len(self.seed_words)

    @property
    def description(self) -> Optional[str]:
        return self._description
