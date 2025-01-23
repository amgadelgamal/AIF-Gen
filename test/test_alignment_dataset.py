import csv
import tempfile

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


@pytest.mark.skip(reason='Need a way to construct AlignmentTask from string')
def test_csv_conversion():
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

    with tempfile.NamedTemporaryFile() as f:
        dataset.to_csv(f.name)
        recovered_dataset = AlignmentDataset.from_csv(f.name)

    assert recovered_dataset.task == dataset.task
    assert recovered_dataset.samples == dataset.samples


def test_from_csv_multiple_tasks_one_csv():
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, 'w', newline='') as f:
            fieldnames = ['task', 'prompt', 'winning_response', 'losing_response']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            first_sample = {
                'task': 'Task #1',
                'prompt': 'Mock prompt',
                'winning_response': 'Winner',
                'losing_response': 'Loser',
            }
            writer.writerow(first_sample)

            second_sample = {
                'task': 'Task #2',  # Task's are not matching
                'prompt': 'Mock prompt',
                'winning_response': 'Winner',
                'losing_response': 'Loser',
            }
            writer.writerow(second_sample)

        with pytest.raises(ValueError):
            _ = AlignmentDataset.from_csv(f.name)


@pytest.mark.skip(reason='Need a way to construct AlignmentTask from string')
def test_json_conversion():
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

    with tempfile.NamedTemporaryFile() as f:
        dataset.to_json(f.name)
        recovered_dataset = AlignmentDataset.from_json(f.name)

    assert recovered_dataset.task == dataset.task
    assert recovered_dataset.samples == dataset.samples
