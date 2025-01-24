import pytest

from aif_gen.dataset.validation import entropy_validation


@pytest.mark.skip(reason='Entropy validation not implemented')
def test_entropy_validation():
    _ = entropy_validation()


@pytest.mark.skip(reason='Entropy validation not implemented')
def test_entropy_countinual_dataset():
    _ = entropy_validation()
