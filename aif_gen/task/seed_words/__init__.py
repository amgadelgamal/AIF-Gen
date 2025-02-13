from typing import List
from .education import EDUCATION_SEED_WORDS
from .finance import FINANCE_SEED_WORDS
from .healthcare import HEALTHCARE_SEED_WORDS
from .politics import POLITICS_SEED_WORDS
from .technology import TECHNOLOGY_SEED_WORDS


def get_seed_words(seed_word_alias: str) -> List[str]:
    """Get the aliased seed word vocabulary for the correspodning seed word alias.

    Args:
        seed_word_alias (str): The seed word alias to get seed word vocabulary for.

    Returns:
        List(str): The list of seed words associated with the seed_word_alias.

    Raises:
        ValueError: If the seed word alias has no known seed word vocabularies.
    """
    seed_word_map = {
        'education': EDUCATION_SEED_WORDS,
        'finance': FINANCE_SEED_WORDS,
        'healthcare': HEALTHCARE_SEED_WORDS,
        'politics': POLITICS_SEED_WORDS,
        'technology': TECHNOLOGY_SEED_WORDS,
    }

    if seed_word_alias not in seed_word_map:
        raise ValueError(
            f'No known seed word vocabulary for {seed_word_alias}, '
            f'expected one of {list(seed_word_map.keys())}'
        )

    return seed_word_map[seed_word_alias]
