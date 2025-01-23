import pytest

from aif_gen.task import AlignmentTask, Domain


def test_init():
    component_dict = {
        'Component A': {
            'seed_words': ['a_foo', 'a_bar', 'a_baz'],
            'description': 'A Mock Domain Component',
        },
        'Component B': {
            'seed_words': ['b_foo', 'b_bar', 'b_baz'],
            'description': 'B Mock Domain Component',
        },
    }
    domain = Domain.from_dict(component_dict)
    objective = 'Mock Objective'
    preference = 'Mock Preference'

    task = AlignmentTask(domain, objective, preference)
    exp_str = f'AlignmentTask({str(domain)}, Objective: Mock Objective, Preference: Mock Preference)'
    assert str(task) == exp_str


def test_init_from_dict():
    task_dict = {
        'domain': {
            'Component A': {
                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                'description': 'A Mock Domain Component',
            },
            'Component B': {
                'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                'description': 'B Mock Domain Component',
            },
        },
        'objective': 'Mock Objective',
        'preference': 'Mock Preference',
    }

    task = AlignmentTask.from_dict(task_dict)
    domain = Domain.from_dict(task_dict['domain'])
    exp_str = f'AlignmentTask({str(domain)}, Objective: Mock Objective, Preference: Mock Preference)'
    assert str(task) == exp_str


def test_init_from_dict_missing_keys():
    task_dict = {  # Missing 'domain' key
        'objective': 'Mock Objective',
        'preference': 'Mock Preference',
    }

    with pytest.raises(ValueError):
        _ = AlignmentTask.from_dict(task_dict)
