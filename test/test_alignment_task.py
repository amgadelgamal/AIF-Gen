from aif_gen.task import AlignmentTask, Domain


def test_init():
    domain = Domain('Mock Domain')
    objective = 'Mock Objective'
    preference = 'Mock Preference'

    task = AlignmentTask(domain, objective, preference)
    assert (
        str(task)
        == 'AlignmentTask(Domain: Mock Domain, Objective: Mock Objective, Preference: Mock Preference)'
    )


def test_init_from_dict():
    pass
