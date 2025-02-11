import asyncio
from unittest.mock import Mock

import pytest

from aif_gen.generate import generate_continual_dataset


def async_mock_return(result):
    fut = asyncio.Future()
    fut.set_result(result)
    return fut


@pytest.fixture
def mock_client():
    mock_obj = Mock()

    mock_obj.chat.completions.create = async_mock_return(
        {'chosen': 'Mock chosen response', 'rejected': 'Mock rejected response'}
    )
    return mock_obj


@pytest.mark.asyncio
@pytest.mark.skip(reason='TODO: Implement data generation tests')
async def test_generate_continual_dataset_schema_parse_error(
    mock_config_dict, mock_client, mock_semaphore
):
    _ = await generate_continual_dataset(mock_config_dict, mock_client, mock_semaphore)


@pytest.mark.asyncio
@pytest.mark.skip(reason='TODO: Implement data generation tests')
async def test_generate_continual_dataset(
    mock_config_dict, mock_client, mock_semaphore
):
    _ = await generate_continual_dataset(mock_config_dict, mock_client, mock_semaphore)
