import pytest

from aif_gen.util.seed import seed_everything


@pytest.fixture(autouse=True)
def run_seed_before_tests():
    seed_everything(1)
    yield


@pytest.fixture(params=[True, False])
def in_place(request):
    return request.param


@pytest.fixture(params=['call', 'apply', 'functional'])
def application_type(request):
    return request.param
