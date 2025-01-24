from typing import Any, Dict, List, Optional


class DomainComponent:
    r"""A domain component is an alias for a set of 'seed words' that describe a
    specific sphere of activity or knowledge.

    Args:
        name (str): The name that describes the content of the domain component.
        seed_words (List[str]): The list of seed words that describe the domain component.
        description (Optional[str]): An optional description of the domain component.

    Raises:
        ValueError: If no seed words are provided.
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

    @classmethod
    def from_dict(cls, component_dict: Dict[str, Any]) -> 'DomainComponent':
        r"""Construct an AlignmentTask from dictionary represenetation.

        Note:
            Expects 'name', and 'seed_words' keys to be present in the dictionary.

        Args:
            component_dict(Dict[str, Any]): The dictionary that encodes the DomainComponent.

        Returns:
            DomainComponent: The newly constructed DomainComponent.

        Raises:
            ValueError: If the input dictionary is missing any required keys.
        """
        expected_keys = {'name', 'seed_words'}
        missing_keys = expected_keys - set(component_dict)
        if len(missing_keys):
            raise ValueError(f'Missing required keys: {missing_keys}')

        name = component_dict['name']
        seed_words = component_dict['seed_words']
        description = component_dict.get('description')
        return cls(name, seed_words, description)

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the DomainComponent to dictionary represenetation.

        Note: This method is the functional inverse of DomainComponent.from_dict().

        Returns:
            Dict[str, Any]: The dictionary representation of the DomainComponent.
        """
        component_dict = {'name': self.name, 'seed_words': self.seed_words}
        if self.description is not None:
            component_dict['description'] = self.description
        return component_dict

    def __str__(self) -> str:
        s = f'{self._name} '
        if self.description is not None:
            s += f'({self.description}) '

        # Truncate number of seed words to first 3 to avoid spamming output stream
        if len(self.seed_words) > 3:
            s += str(self._seed_words[:3])[:-1] + ', ...]'
        else:
            s += str(self._seed_words)

        return s

    @property
    def name(self) -> str:
        """str: The name of this DomainComponent."""
        return self._name

    @property
    def seed_words(self) -> List[str]:
        """List[str]: The list of seed words aliased by this DomainComponent."""
        return self._seed_words

    @property
    def num_seed_words(self) -> int:
        """int: The number of seed words aliased by this DomainComponent."""
        return len(self.seed_words)

    @property
    def description(self) -> Optional[str]:
        """Optional[str]: The description in the current DomainComponent, if it exists."""
        return self._description
