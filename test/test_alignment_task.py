from aif_gen.task import AlignmentTask, Domain, Objective, Preference


def test_init():
    domain = Domain()
    objective = Objective()
    preference = Preference()

    task = AlignmentTask(domain, objective, preference)
    assert (
        str(task)
        == 'Domain: Mock Domain, Objective: Mock Objective, Preference: Mock Preference'
    )
