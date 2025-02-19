import tempfile

from datasets import Dataset

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.task import AlignmentTask, Domain, DomainComponent


def test_init_():
    datasets = [_generate_dataset(i) for i in range(3)]
    ca_dataset = ContinualAlignmentDataset(datasets)

    assert ca_dataset.datasets == datasets
    assert ca_dataset.num_datasets == len(datasets)
    assert ca_dataset.num_samples == sum(len(dataset) for dataset in datasets)
    assert len(ca_dataset) == sum(len(dataset) for dataset in datasets)


def test_append():
    datasets = [_generate_dataset(i) for i in range(3)]
    ca_dataset = ContinualAlignmentDataset([])

    for i in range(len(datasets)):
        assert ca_dataset.datasets == datasets[:i]
        assert ca_dataset.num_datasets == i
        assert ca_dataset.num_samples == sum(len(dataset) for dataset in datasets[:i])
        assert len(ca_dataset) == sum(len(dataset) for dataset in datasets[:i])
        ca_dataset.append(datasets[i])


def test_extend():
    datasets = [_generate_dataset(i) for i in range(3)]
    ca_dataset = ContinualAlignmentDataset([])

    assert ca_dataset.datasets == []
    assert ca_dataset.num_datasets == 0
    assert len(ca_dataset) == 0

    ca_dataset.extend(datasets)

    assert ca_dataset.datasets == datasets
    assert ca_dataset.num_datasets == len(datasets)
    assert ca_dataset.num_samples == sum(len(dataset) for dataset in datasets)
    assert len(ca_dataset) == sum(len(dataset) for dataset in datasets)


def test_slice():
    datasets = [_generate_dataset(i) for i in range(3)]
    ca_dataset = ContinualAlignmentDataset(datasets)

    all_samples = []
    for dataset in ca_dataset.datasets:
        all_samples.extend(dataset.samples)
    assert ca_dataset[0] == all_samples[0]
    assert ca_dataset[-1] == all_samples[-1]
    assert ca_dataset[1:2] == all_samples[1:2]
    assert ca_dataset[2:7] == all_samples[2:7]
    assert ca_dataset[:] == all_samples[:]


def test_dict_conversion():
    datasets = [_generate_dataset(i) for i in range(3)]
    ca_dataset = ContinualAlignmentDataset(datasets)

    dataset_dict = ca_dataset.to_dict()
    recovered_ca_dataset = ContinualAlignmentDataset.from_dict(dataset_dict)

    assert ca_dataset.num_datasets == recovered_ca_dataset.num_datasets
    assert ca_dataset.num_samples == recovered_ca_dataset.num_samples
    assert len(ca_dataset) == len(recovered_ca_dataset)

    for i, dataset in enumerate(recovered_ca_dataset.datasets):
        assert str(ca_dataset.datasets[i].task.domain) == str(dataset.task.domain)
        assert ca_dataset.datasets[i].task.objective == dataset.task.objective
        assert ca_dataset.datasets[i].task.preference == dataset.task.preference
        assert ca_dataset.datasets[i].samples == dataset.samples


def test_json_conversion():
    datasets = [_generate_dataset(i) for i in range(3)]
    ca_dataset = ContinualAlignmentDataset(datasets)

    with tempfile.NamedTemporaryFile() as f:
        ca_dataset.to_json(f.name)
        recovered_ca_dataset = ContinualAlignmentDataset.from_json(f.name)

    assert ca_dataset.num_datasets == recovered_ca_dataset.num_datasets
    assert ca_dataset.num_samples == recovered_ca_dataset.num_samples
    assert len(ca_dataset) == len(recovered_ca_dataset)

    for i, dataset in enumerate(recovered_ca_dataset.datasets):
        assert str(ca_dataset.datasets[i].task.domain) == str(dataset.task.domain)
        assert ca_dataset.datasets[i].task.objective == dataset.task.objective
        assert ca_dataset.datasets[i].task.preference == dataset.task.preference
        assert ca_dataset.datasets[i].samples == dataset.samples


def test_hf_compatiblity():
    datasets = [_generate_dataset(i) for i in range(3)]
    ca_dataset = ContinualAlignmentDataset(datasets)

    list_of_dicts = ca_dataset.to_hf_compatible()

    # only ensure the type is list[dict[str, Dataset]]
    # the rest is tested in test_alignment_dataset.py
    assert isinstance(list_of_dicts, list)
    assert all(isinstance(d, dict) for d in list_of_dicts)
    assert all('train' in d for d in list_of_dicts)
    assert all('test' in d for d in list_of_dicts)
    assert all(isinstance(d['train'], Dataset) for d in list_of_dicts)


def _generate_dataset(dataset_id: int) -> AlignmentDataset:
    component_a = DomainComponent('Component A', ['a_foo', 'a_bar', 'a_baz'])
    component_b = DomainComponent('Component B', ['b_foo', 'b_bar'])
    component_c = DomainComponent('Component C', ['c_foo', 'c_bar', 'c_baz', 'c_bat'])
    components = [component_a, component_b, component_c]
    weights = [1.5, 0.7, 0.3]  # Need not be normalized

    domain = Domain(components, weights)
    objective = f'Mock Objective {dataset_id}'
    preference = f'Mock Preference {dataset_id}'
    task = AlignmentTask(domain, objective, preference)

    samples = [
        AlignmentDatasetSample(
            f'Mock prompt A {dataset_id}',
            f'Winning Response A {dataset_id}',
            f'Losing Response A {dataset_id}',
        ),
        AlignmentDatasetSample(
            f'Mock prompt B {dataset_id}',
            f'Winning Response B {dataset_id}',
            f'Losing Response B {dataset_id}',
        ),
        AlignmentDatasetSample(
            f'Mock prompt C {dataset_id}',
            f'Winning Response C {dataset_id}',
            f'Losing Response C {dataset_id}',
        ),
    ]

    dataset = AlignmentDataset(task, samples)
    return dataset
