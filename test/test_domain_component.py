import pytest

from aif_gen.task import DomainComponent
from aif_gen.task.seed_words import get_seed_words


def test_init():
    name = 'TestComponent'
    seed_words = ['foo', 'bar', 'baz']
    description = 'Mock Domain Component'

    component = DomainComponent(name, seed_words, description)
    assert component.name == name
    assert component.seed_words == seed_words
    assert component.description == description
    assert component.num_seed_words == len(seed_words)


def test_init_no_description():
    name = 'TestComponent'
    seed_words = ['foo', 'bar', 'baz']

    component = DomainComponent(name, seed_words)
    assert component.name == name
    assert component.seed_words == seed_words
    assert component.description is None
    assert component.num_seed_words == len(seed_words)


def test_init_empty_seed_words():
    name = 'BadComponent'
    seed_words = []
    with pytest.raises(ValueError):
        _ = DomainComponent(name, seed_words)


@pytest.mark.parametrize(
    'seed_word_alias', ['education', 'finance', 'healthcare', 'politics', 'technology']
)
def test_init_seed_word_alias(seed_word_alias):
    name = 'TestComponent'
    description = 'Mock Domain Component'

    component = DomainComponent(name, seed_word_alias, description)
    exp_seed_words = get_seed_words(seed_word_alias)
    assert component.name == name
    assert component.seed_words == exp_seed_words
    assert component.description == description
    assert component.num_seed_words == len(exp_seed_words)


def test_init_from_dict():
    component_dict = {
        'name': 'TestComponent',
        'seed_words': ['foo', 'bar', 'baz'],
        'description': 'Mock Domain Component',
    }

    component = DomainComponent.from_dict(component_dict)
    assert component.name == component_dict['name']
    assert component.seed_words == component_dict['seed_words']
    assert component.description == component_dict['description']
    assert component.num_seed_words == len(component_dict['seed_words'])


def test_init_from_dict_no_description():
    component_dict = {
        'name': 'TestComponent',
        'seed_words': ['foo', 'bar', 'baz'],
    }

    component = DomainComponent.from_dict(component_dict)
    assert component.name == component_dict['name']
    assert component.seed_words == component_dict['seed_words']
    assert component.description is None
    assert component.num_seed_words == len(component_dict['seed_words'])


def test_init_from_dict_missing_keys():
    component_dict = {  # Missing 'seed_words' key
        'name': 'BadComponent',
    }

    with pytest.raises(ValueError):
        _ = DomainComponent.from_dict(component_dict)


def test_init_to_dict():
    component_dict = {
        'name': 'TestComponent',
        'seed_words': ['foo', 'bar', 'baz'],
        'description': 'Mock Domain Component',
    }

    component = DomainComponent.from_dict(component_dict)
    assert component_dict == component.to_dict()
