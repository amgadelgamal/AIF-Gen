import pytest


@pytest.fixture(params=[0, 0.3, 0.5, 1])
def train_frac(request):
    return request.param
