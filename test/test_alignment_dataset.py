import pytest

from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample
from aif_gen.task import AlignmentTask, Domain


def test_init_():
    domain = Domain('Mock Domain')
    objective = 'Mock Objective'
    preference = 'Mock Preference'
    task = AlignmentTask(domain, objective, preference)

    samples = [
        AlignmentDatasetSample(
            'Mock prompt A', 'Winning Response A', 'Losing Response A'
        ),
        AlignmentDatasetSample(
            'Mock prompt B', 'Winning Response B', 'Losing Response B'
        ),
        AlignmentDatasetSample(
            'Mock prompt C', 'Winning Response C', 'Losing Response C'
        ),
    ]

    dataset = AlignmentDataset(task, samples)
    assert dataset.task == task
    assert dataset.samples == samples
    assert len(dataset) == len(samples)


def test_append():
    domain = Domain('Mock Domain')
    objective = 'Mock Objective'
    preference = 'Mock Preference'
    task = AlignmentTask(domain, objective, preference)

    samples = [
        AlignmentDatasetSample(
            'Mock prompt A', 'Winning Response A', 'Losing Response A'
        ),
        AlignmentDatasetSample(
            'Mock prompt B', 'Winning Response B', 'Losing Response B'
        ),
        AlignmentDatasetSample(
            'Mock prompt C', 'Winning Response C', 'Losing Response C'
        ),
    ]

    dataset = AlignmentDataset(task, [])

    for i in range(len(samples)):
        assert dataset.task == task
        assert dataset.samples == samples[:i]
        assert len(dataset) == i
        dataset.append(samples[i])


def test_append_bad_type():
    domain = Domain('Mock Domain')
    objective = 'Mock Objective'
    preference = 'Mock Preference'
    task = AlignmentTask(domain, objective, preference)

    dataset = AlignmentDataset(task, [])
    with pytest.raises(TypeError):
        dataset.append('Bad type')


def test_extend():
    domain = Domain('Mock Domain')
    objective = 'Mock Objective'
    preference = 'Mock Preference'
    task = AlignmentTask(domain, objective, preference)

    samples = [
        AlignmentDatasetSample(
            'Mock prompt A', 'Winning Response A', 'Losing Response A'
        ),
        AlignmentDatasetSample(
            'Mock prompt B', 'Winning Response B', 'Losing Response B'
        ),
        AlignmentDatasetSample(
            'Mock prompt C', 'Winning Response C', 'Losing Response C'
        ),
    ]

    dataset = AlignmentDataset(task, [])
    dataset.extend(samples)

    assert dataset.task == task
    assert dataset.samples == samples
    assert len(dataset) == len(samples)


def test_slice():
    domain = Domain('Mock Domain')
    objective = 'Mock Objective'
    preference = 'Mock Preference'
    task = AlignmentTask(domain, objective, preference)

    samples = [
        AlignmentDatasetSample(
            'Mock prompt A', 'Winning Response A', 'Losing Response A'
        ),
        AlignmentDatasetSample(
            'Mock prompt B', 'Winning Response B', 'Losing Response B'
        ),
        AlignmentDatasetSample(
            'Mock prompt C', 'Winning Response C', 'Losing Response C'
        ),
    ]

    dataset = AlignmentDataset(task, samples)
    assert dataset[0] == samples[0]
    assert dataset[-1] == samples[-1]
    assert dataset[1:2] == samples[1:2]
    assert dataset[:] == samples[:]


def test_to_csv():
    pass


def test_from_csv():
    pass


def test_from_json():
    pass


def test_to_json():
    pass
