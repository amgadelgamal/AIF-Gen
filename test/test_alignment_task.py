import pytest

from aif_gen.task import AlignmentTask, Domain


def test_init():
    domain = Domain('Mock Domain')
    objective = 'Mock Objective'
    preference = 'Mock Preference'

    task = AlignmentTask(domain, objective, preference)
    exp_str = 'AlignmentTask(Domain: Mock Domain, Objective: Mock Objective, Preference: Mock Preference)'
    assert str(task) == exp_str


def test_init_from_dict():
    task_dict = {
        'domain': 'Mock Domain',
        'objective': 'Mock Objective',
        'preference': 'Mock Preference',
    }

    task = AlignmentTask.from_dict(task_dict)
    exp_str = 'AlignmentTask(Domain: Mock Domain, Objective: Mock Objective, Preference: Mock Preference)'
    assert str(task) == exp_str


def test_init_from_dict_missing_keys():
    task_dict = {  # Missing 'preference' key
        'domain': 'Mock Domain',
        'objective': 'Mock Objective',
    }

    with pytest.raises(ValueError):
        _ = AlignmentTask.from_dict(task_dict)


@pytest.mark.skip(reason='Not implemented, waiting for Domain PR')
def test_to_dict():
    task_dict = {
        'domain': 'Mock Domain',
        'objective': 'Mock Objective',
        'preference': 'Mock Preference',
    }

    task = AlignmentTask.from_dict(task_dict)
    assert task.to_dict() == task_dict
