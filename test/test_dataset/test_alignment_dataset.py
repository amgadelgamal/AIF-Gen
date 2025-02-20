import tempfile

import pytest
from datasets import Dataset

from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample
from aif_gen.task import AlignmentTask, Domain


def test_init(train_frac):
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

    dataset = AlignmentDataset(task, samples, train_frac)
    assert dataset.task == task
    assert dataset.samples == samples
    assert dataset.num_samples == len(samples)
    assert len(dataset) == len(samples)
    assert dataset.train_frac == train_frac
    assert dataset.test_frac == 1.0 - train_frac
    assert dataset.num_train_samples == int(len(samples) * dataset.train_frac)
    assert dataset.num_test_samples == len(samples) - dataset.num_train_samples
    assert dataset.train == samples[: dataset.num_train_samples]
    assert dataset.test == samples[dataset.num_train_samples :]


@pytest.mark.parametrize('bad_train_frac', [-1, -0.5, 1.5])
def test_init_bad_train_frac(bad_train_frac):
    mock_task = None
    mock_samples = []
    with pytest.raises(ValueError):
        _ = AlignmentDataset(mock_task, mock_samples, bad_train_frac)


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


def test_dict_conversion(train_frac):
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

    dataset = AlignmentDataset(task, samples, train_frac)

    dataset_dict = dataset.to_dict()
    recovered_dataset = AlignmentDataset.from_dict(dataset_dict)

    assert str(recovered_dataset.task.domain) == str(domain)
    assert recovered_dataset.task.objective == objective
    assert recovered_dataset.task.preference == preference
    assert recovered_dataset.samples == dataset.samples


def test_json_conversion(train_frac):
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

    dataset = AlignmentDataset(task, samples, train_frac)

    with tempfile.NamedTemporaryFile() as f:
        dataset.to_json(f.name)
        recovered_dataset = AlignmentDataset.from_json(f.name)

    assert str(recovered_dataset.task.domain) == str(domain)
    assert recovered_dataset.task.objective == objective
    assert recovered_dataset.task.preference == preference
    assert recovered_dataset.samples == dataset.samples


def test_hf_compatiblity(train_frac):
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

    dataset = AlignmentDataset(task, samples, train_frac)

    hf_dict = dataset.to_hf_compatible()

    assert type(hf_dict) == dict
    assert 'train' in hf_dict
    assert 'test' in hf_dict
    assert type(hf_dict['train']) == Dataset

    assert hf_dict['train']['prompt'] == [sample.prompt for sample in dataset.train]
    assert hf_dict['train']['chosen'] == [sample.chosen for sample in dataset.train]
    assert hf_dict['train']['rejected'] == [sample.rejected for sample in dataset.train]

    assert hf_dict['test']['prompt'] == [sample.prompt for sample in dataset.test]
    assert hf_dict['test']['chosen'] == [sample.chosen for sample in dataset.test]
    assert hf_dict['test']['rejected'] == [sample.rejected for sample in dataset.test]
