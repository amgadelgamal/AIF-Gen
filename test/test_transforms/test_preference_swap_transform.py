import pytest

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.dataset.transforms import PreferenceSwapTransform
from aif_gen.dataset.transforms import functional as F


@pytest.fixture(params=['call', 'apply', 'functional'])
def application_type(request):
    return request.param


@pytest.mark.parametrize('bad_swap_probability', [-1, 2])
def test_apply_preference_swap_bad_swap_probability(bad_swap_probability):
    with pytest.raises(ValueError):
        _ = PreferenceSwapTransform(swap_probability=bad_swap_probability)

    # Check failure when using the setter method
    transform = PreferenceSwapTransform(swap_probability=0.3)
    with pytest.raises(ValueError):
        transform.swap_probability = bad_swap_probability

    # Check failure when using the functional API
    mock_dataset = None
    with pytest.raises(ValueError):
        F.preference_swap_transform(mock_dataset, swap_probability=bad_swap_probability)


def mock_task():
    return {
        'domain': {
            'Component A': {
                'name': 'Component A',
                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                'weight': 0.5,
            },
            'Component B': {
                'name': 'Component B',
                'seed_words': ['b_foo', 'b_bar'],
                'weight': 0.5,
            },
        },
        'objective': 'Mock Objective 1',
        'preference': 'Mock Preference 1',
    }


def mock_dataset_dict(continual):
    if continual:
        return {
            'datasets': [
                {
                    'task': mock_task(),
                    'samples': [
                        {
                            'prompt': 'Mock prompt A 1',
                            'chosen': 'Winning Response A 1',
                            'rejected': 'Losing Response A 1',
                        },
                        {
                            'prompt': 'Mock prompt B 1',
                            'chosen': 'Winning Response B 1',
                            'rejected': 'Losing Response B 1',
                        },
                        {
                            'prompt': 'Mock prompt C 1',
                            'chosen': 'Winning Response C 1',
                            'rejected': 'Losing Response C 1',
                        },
                    ],
                },
                {
                    'task': mock_task(),
                    'samples': [
                        {
                            'prompt': 'Mock prompt A 2',
                            'chosen': 'Winning Response A 2',
                            'rejected': 'Losing Response A 2',
                        },
                        {
                            'prompt': 'Mock prompt B 2',
                            'chosen': 'Winning Response B 2',
                            'rejected': 'Losing Response B 2',
                        },
                        {
                            'prompt': 'Mock prompt C 2',
                            'chosen': 'Winning Response C 2',
                            'rejected': 'Losing Response C 2',
                        },
                    ],
                },
            ]
        }
    else:
        return {
            'task': mock_task(),
            'samples': [
                {
                    'prompt': 'Mock prompt A 1',
                    'chosen': 'Winning Response A 1',
                    'rejected': 'Losing Response A 1',
                },
                {
                    'prompt': 'Mock prompt B 1',
                    'chosen': 'Winning Response B 1',
                    'rejected': 'Losing Response B 1',
                },
                {
                    'prompt': 'Mock prompt C 1',
                    'chosen': 'Winning Response C 1',
                    'rejected': 'Losing Response C 1',
                },
            ],
        }


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=False)])
def test_apply_preference_swap_to_static_dataset(
    dataset_dict, in_place, application_type
):
    expected_dataset_dict = {
        'task': mock_task(),
        'samples': [
            {
                'prompt': 'Mock prompt A 1',
                'chosen': 'Winning Response A 1',
                'rejected': 'Losing Response A 1',
            },
            {
                'prompt': 'Mock prompt B 1',
                'chosen': 'Losing Response B 1',
                'rejected': 'Winning Response B 1',
            },
            {
                'prompt': 'Mock prompt C 1',
                'chosen': 'Winning Response C 1',
                'rejected': 'Losing Response C 1',
            },
        ],
    }

    dataset = AlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=0.5, in_place=in_place
        )

    assert transformed_dataset.to_dict()['samples'] == expected_dataset_dict['samples']
    if in_place:
        assert dataset == transformed_dataset


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=False)])
def test_apply_preference_swap_to_static_dataset_full_swap(
    in_place, application_type, dataset_dict
):
    expected_dataset_dict = {
        'task': mock_task(),
        'samples': [
            {
                'prompt': 'Mock prompt A 1',
                'chosen': 'Losing Response A 1',
                'rejected': 'Winning Response A 1',
            },
            {
                'prompt': 'Mock prompt B 1',
                'chosen': 'Losing Response B 1',
                'rejected': 'Winning Response B 1',
            },
            {
                'prompt': 'Mock prompt C 1',
                'chosen': 'Losing Response C 1',
                'rejected': 'Winning Response C 1',
            },
        ],
    }

    dataset = AlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 1  # Setter overrides swap probability

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=1, in_place=in_place
        )

    assert transformed_dataset.to_dict()['samples'] == expected_dataset_dict['samples']
    if in_place:
        assert dataset == transformed_dataset


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=True)])
def test_apply_preference_swap_to_continual_dataset_full_swap(
    in_place, application_type, dataset_dict
):
    expected_dataset_dict = {
        'datasets': [
            {
                'task': mock_task(),
                'samples': [
                    {
                        'prompt': 'Mock prompt A 1',
                        'chosen': 'Losing Response A 1',
                        'rejected': 'Winning Response A 1',
                    },
                    {
                        'prompt': 'Mock prompt B 1',
                        'chosen': 'Losing Response B 1',
                        'rejected': 'Winning Response B 1',
                    },
                    {
                        'prompt': 'Mock prompt C 1',
                        'chosen': 'Losing Response C 1',
                        'rejected': 'Winning Response C 1',
                    },
                ],
            },
            {
                'task': mock_task(),
                'samples': [
                    {
                        'prompt': 'Mock prompt A 2',
                        'chosen': 'Losing Response A 2',
                        'rejected': 'Winning Response A 2',
                    },
                    {
                        'prompt': 'Mock prompt B 2',
                        'chosen': 'Losing Response B 2',
                        'rejected': 'Winning Response B 2',
                    },
                    {
                        'prompt': 'Mock prompt C 2',
                        'chosen': 'Losing Response C 2',
                        'rejected': 'Winning Response C 2',
                    },
                ],
            },
        ]
    }

    dataset = ContinualAlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 1  # Setter overrides swap probability

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=1, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=False)])
def test_apply_preference_swap_to_static_dataset_no_swap(
    in_place, application_type, dataset_dict
):
    expected_dataset_dict = dataset_dict
    dataset = AlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 0  # Setter overrides swap probability

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=0, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=True)])
def test_apply_preference_swap_to_continual_dataset_no_swap(
    in_place, application_type, dataset_dict
):
    expected_dataset_dict = dataset_dict
    dataset = ContinualAlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 0  # Setter overrides swap probability

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=0, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset
