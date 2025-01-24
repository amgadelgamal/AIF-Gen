import pytest

from aif_gen.dataset.validation import drift_detection


@pytest.mark.skip(reason='Drift detection not implemented')
def test_drift_detection():
    _ = drift_detection()
