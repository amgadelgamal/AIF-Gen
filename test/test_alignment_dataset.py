import tempfile

from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample
from aif_gen.task import AlignmentTask, Domain


def test_init_():
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
    assert dataset.num_samples == len(samples)
    assert len(dataset) == len(samples)


def test_slice():
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


def test_dict_conversion():
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

    dataset_dict = dataset.to_dict()
    recovered_dataset = AlignmentDataset.from_dict(dataset_dict)

    assert str(recovered_dataset.task.domain) == str(domain)
    assert recovered_dataset.task.objective == objective
    assert recovered_dataset.task.preference == preference
    assert recovered_dataset.samples == dataset.samples


def test_json_conversion():
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

    assert str(recovered_dataset.task.domain) == str(domain)
    assert recovered_dataset.task.objective == objective
    assert recovered_dataset.task.preference == preference
    assert recovered_dataset.samples == dataset.samples
