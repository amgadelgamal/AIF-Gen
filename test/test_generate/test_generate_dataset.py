import asyncio
import json

import openai
import pytest

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.dataset.alignment_sample import AlignmentDatasetSample
from aif_gen.generate import generate_continual_dataset
from aif_gen.task import AlignmentTask


def async_mock_return(result):
    fut = asyncio.Future()
    fut.set_result(result)
    return fut


@pytest.fixture
def mock_client(mocker):
    mock_response = mocker.MagicMock(name='response')
    mock_response.choices[0].message.content = json.dumps(
        {
            'chosen': 'Mock chosen response',
            'rejected': 'Mock rejected response',
        }
    )
    mock_client = mocker.MagicMock(name='client')
    mock_client.chat.completions.create.return_value = async_mock_return(mock_response)
    return mock_client


@pytest.fixture
def mock_client_openai_exception(mocker):
    mock_response = mocker.MagicMock(name='response')
    mock_response.choices[0].message.content = json.dumps(
        {
            'chosen': 'Mock chosen response',
            'rejected': 'Mock rejected response',
        }
    )
    mock_client = mocker.MagicMock(name='client')
    mock_client.chat.completions.create.side_effect = openai.NotFoundError(
        response=mock_response, body='mock', message='mock'
    )
    return mock_client


@pytest.fixture(
    params=[
        {'chosen': 'Mock chosen response'},  # Missing the 'rejected' response
        {'rejected': 'Mock rejected response'},  # Missing the 'chosen' response
        {'Foo Bar': 'Baz'},  # 'Missing both 'chosen' and 'rejected' keys
        'Foo Bar Baz',  # Not even a dictionary
    ]
)
def mock_client_schema_parse_exception(mocker, request):
    # Fail the first sample schema parse, then switch to good schema, and ensure the other samples are generated
    mock_bad_response = mocker.MagicMock(name='response')
    mock_bad_response.choices[0].message.content = json.dumps(request.param)

    mock_good_response = mocker.MagicMock(name='response')
    mock_good_response.choices[0].message.content = json.dumps(
        {
            'chosen': 'Mock chosen response',
            'rejected': 'Mock rejected response',
        }
    )

    mock_client = mocker.MagicMock(name='client')

    # This is a crude way of making the mock return a 'bad response' the first 5 times,
    # then 'good response' for the next 100 calls. 100 is arbitrary, but it should be
    # larger than the mock dataset we test here. If we do run on more than 100 samples,
    # this will raise a StopIteration exception, causes a flaky test. The proper way to
    # mock this would involve looking at the calling args, since the first API call doesn't
    # actually check for json schema.
    mock_client.chat.completions.create.side_effect = 5 * [
        async_mock_return(mock_bad_response)
    ] + 100 * [async_mock_return(mock_good_response)]

    return mock_client


@pytest.fixture
def mock_semaphore(mocker):
    mock_sem = mocker.MagicMock(name='async_semaphore')
    mock_sem.return_value.__aenter__.return_value = None
    return mock_sem


@pytest.fixture
def mock_config_dict():
    return {
        'model_name': 'mock',
        'data': {
            'task_specs': [
                {
                    'num_samples': 5,
                    'alignment_task': {
                        'domain': {
                            'Component A': {
                                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                            },
                            'Component B': {
                                'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                            },
                        },
                        'objective': 'Mock Objective 1',
                        'preference': 'Mock Preference 1',
                    },
                },
                {
                    'num_samples': 5,
                    'alignment_task': {
                        'domain': {
                            'Component A': {
                                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                            },
                            'Component B': {
                                'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                            },
                        },
                        'objective': 'Mock Objective 2',
                        'preference': 'Mock Preference 2',
                    },
                },
            ]
        },
    }


@pytest.mark.asyncio
async def test_generate_continual_dataset_schema_parse_exception(
    mock_config_dict, mock_client_schema_parse_exception, mock_semaphore
):
    continual_dataset = await generate_continual_dataset(
        mock_config_dict, mock_client_schema_parse_exception, mock_semaphore
    )

    task_specs = mock_config_dict['data']['task_specs']
    assert isinstance(continual_dataset, ContinualAlignmentDataset)
    assert len(task_specs) == len(continual_dataset.datasets)

    for i in range(len(continual_dataset.datasets)):
        dataset = continual_dataset.datasets[i]
        assert isinstance(dataset, AlignmentDataset)

        exp_task = AlignmentTask.from_dict(task_specs[i]['alignment_task'])

        assert len(dataset) > 0  # TODO: This is flaky
        assert dataset.task.to_dict() == exp_task.to_dict()

        for sample in dataset.samples:
            assert isinstance(sample, AlignmentDatasetSample)


@pytest.mark.asyncio
@pytest.mark.skip('TODO: Proper cleanup')
async def test_generate_continual_dataset_uncaught_exception(
    mock_config_dict, mock_client_openai_exception, mock_semaphore
):
    with pytest.raises(openai.NotFoundError):
        continual_dataset = await generate_continual_dataset(
            mock_config_dict, mock_client_openai_exception, mock_semaphore
        )
        assert continual_dataset is None
        # assert not mock_client.called  # Cannot make this assertion since some requests might have been attempted


@pytest.mark.asyncio
@pytest.mark.parametrize('pop_key', ['model_name', 'data'])
async def test_generate_continual_dataset_bad_config_dict(
    mock_config_dict,
    mock_client,
    mock_semaphore,
    pop_key,
):
    mock_config_dict.pop(pop_key)  # Remove required key
    with pytest.raises(KeyError):
        continual_dataset = await generate_continual_dataset(
            mock_config_dict, mock_client, mock_semaphore
        )
        assert continual_dataset is None
        assert not mock_client.called  # Make sure we didn't hit the model endpoint


@pytest.mark.asyncio
async def test_generate_continual_dataset(
    mock_config_dict, mock_client, mock_semaphore
):
    continual_dataset = await generate_continual_dataset(
        mock_config_dict, mock_client, mock_semaphore
    )
    task_specs = mock_config_dict['data']['task_specs']
    assert isinstance(continual_dataset, ContinualAlignmentDataset)
    assert len(task_specs) == len(continual_dataset.datasets)

    for i in range(len(continual_dataset.datasets)):
        dataset = continual_dataset.datasets[i]
        assert isinstance(dataset, AlignmentDataset)

        exp_num_samples = task_specs[i]['num_samples']
        exp_task = AlignmentTask.from_dict(task_specs[i]['alignment_task'])

        assert len(dataset) == exp_num_samples
        assert dataset.task.to_dict() == exp_task.to_dict()

        for sample in dataset.samples:
            assert isinstance(sample, AlignmentDatasetSample)
