import pytest

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.dataset.transforms import PreferenceSwapTransform
from aif_gen.util.seed import seed_everything


@pytest.fixture(autouse=True)
def run_seed_before_tests():
    seed_everything(1)
    yield


def test_apply_preference_swap_bad_swap_probability():
    with pytest.raises(ValueError):
        _ = PreferenceSwapTransform(swap_probability=-1)

    with pytest.raises(ValueError):
        _ = PreferenceSwapTransform(swap_probability=2)

    # Check failure when using the setter as well
    transform = PreferenceSwapTransform(swap_probability=0.3)

    with pytest.raises(ValueError):
        transform.swap_probability = -1

    with pytest.raises(ValueError):
        transform.swap_probability = 2


def test_apply_preference_swap_to_static_dataset():
    dataset_dict = {
        'task': {
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
        },
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

    expected_dataset_dict = {
        'task': {
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
        },
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

    assert transform(dataset).to_dict() == expected_dataset_dict


def test_apply_preference_swap_to_static_dataset_full_swap():
    dataset_dict = {
        'task': {
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
        },
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

    expected_dataset_dict = {
        'task': {
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
        },
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

    assert transform(dataset).to_dict() == expected_dataset_dict
    assert transform.apply(dataset).to_dict() == expected_dataset_dict


def test_apply_preference_swap_to_continual_dataset_full_swap():
    dataset_dict = {
        'datasets': [
            {
                'task': {
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
                },
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
                'task': {
                    'domain': {
                        'Component C': {
                            'name': 'Component C',
                            'seed_words': ['c_foo', 'c_bar', 'c_baz'],
                            'weight': 0.7,
                        },
                        'Component D': {
                            'name': 'Component D',
                            'seed_words': ['d_foo', 'd_bar'],
                            'weight': 0.3,
                        },
                    },
                    'objective': 'Mock Objective 2',
                    'preference': 'Mock Preference 2',
                },
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

    expected_dataset_dict = {
        'datasets': [
            {
                'task': {
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
                },
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
                'task': {
                    'domain': {
                        'Component C': {
                            'name': 'Component C',
                            'seed_words': ['c_foo', 'c_bar', 'c_baz'],
                            'weight': 0.7,
                        },
                        'Component D': {
                            'name': 'Component D',
                            'seed_words': ['d_foo', 'd_bar'],
                            'weight': 0.3,
                        },
                    },
                    'objective': 'Mock Objective 2',
                    'preference': 'Mock Preference 2',
                },
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

    assert transform(dataset).to_dict() == expected_dataset_dict
    assert transform.apply(dataset).to_dict() == expected_dataset_dict


def test_apply_preference_swap_to_static_dataset_no_swap():
    dataset_dict = {
        'task': {
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
        },
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

    expected_dataset_dict = dataset_dict

    dataset = AlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 0  # Setter overrides swap probability

    assert transform(dataset).to_dict() == expected_dataset_dict
    assert transform.apply(dataset).to_dict() == expected_dataset_dict


def test_apply_preference_swap_to_continual_dataset_no_swap():
    dataset_dict = {
        'datasets': [
            {
                'task': {
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
                },
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
                'task': {
                    'domain': {
                        'Component C': {
                            'name': 'Component C',
                            'seed_words': ['c_foo', 'c_bar', 'c_baz'],
                            'weight': 0.7,
                        },
                        'Component D': {
                            'name': 'Component D',
                            'seed_words': ['d_foo', 'd_bar'],
                            'weight': 0.3,
                        },
                    },
                    'objective': 'Mock Objective 2',
                    'preference': 'Mock Preference 2',
                },
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

    expected_dataset_dict = dataset_dict

    dataset = ContinualAlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 0  # Setter overrides swap probability

    assert transform(dataset).to_dict() == expected_dataset_dict
    assert transform.apply(dataset).to_dict() == expected_dataset_dict
