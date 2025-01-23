import pytest

from aif_gen.task import DomainComponent


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
