import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.dataset.validation import CoherenceEvaluator


class DummyTokenizer:
    def __call__(self, text, return_tensors='pt'):
        # Return a dummy encoding with a dummy input_ids tensor.
        return type('DummyEncodings', (), {'input_ids': torch.tensor([[1, 2, 3]])})


class DummyModel:
    def eval(self):
        pass

    def __call__(self, input_ids, labels):
        # Return a dummy output with a constant loss of 1.0.
        # This gives a perplexity of exp(1) ≈ 2.71828 and a coherence score of ≈ 1/2.71828 ≈ 0.3679.
        # Multiplied by 100 and rounded, we expect 37.
        return type('DummyOutput', (), {'loss': torch.tensor(1.0)})


# Helper Function to Patch the Evaluator's Dependencies ---
@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch):
    # Patch the from_pretrained class methods to return our dummy tokenizer and model.
    monkeypatch.setattr(
        GPT2TokenizerFast, 'from_pretrained', lambda model_name: DummyTokenizer()
    )
    monkeypatch.setattr(
        GPT2LMHeadModel, 'from_pretrained', lambda model_name: DummyModel()
    )


@pytest.mark.skip('TODO: Patch the classifier to avoid downloaded when running tests')
def test_evaluate_single_dataset():
    """Test that evaluate() returns a dictionary mapping sample IDs to coherence scores
    for a single AlignmentDataset.
    """
    evaluator = CoherenceEvaluator()
    # Create two dummy samples. (No 'id' attribute is provided, so fallback ids are used.)
    sample1 = AlignmentDatasetSample(
        prompt='Prompt 1', chosen='Chosen text 1', rejected='Rejected text'
    )
    sample2 = AlignmentDatasetSample(
        prompt='Prompt 2', chosen='Chosen text 2', rejected='Rejected text'
    )
    dataset = AlignmentDataset(task=None, samples=[sample1, sample2])

    scores = evaluator.evaluate(dataset)
    # Since our dummy model always returns a loss of 1.0,
    # we expect coherence score = int(round((1/exp(1))*100)) = 37 for each sample.
    expected = {'0': 37, '1': 37}
    assert scores == expected


@pytest.mark.skip('TODO: Patch the classifier to avoid downloaded when running tests')
def test_coherence_evaluation_single_dataset():
    """Test that coherence_evaluation() returns a one-element list containing
    the evaluation dict for a single AlignmentDataset.
    """
    # Create two dummy samples.
    sample1 = AlignmentDatasetSample(
        prompt='Prompt 1', chosen='Chosen text 1', rejected='Rejected text'
    )
    sample2 = AlignmentDatasetSample(
        prompt='Prompt 2', chosen='Chosen text 2', rejected='Rejected text'
    )
    dataset = AlignmentDataset(task=None, samples=[sample1, sample2])

    evaluator = CoherenceEvaluator()
    results = evaluator.coherence_evaluation(dataset)
    # Expect a list with one dict mapping "0" and "1" to 37.
    expected = [{'0': 37, '1': 37}]
    assert results == expected


@pytest.mark.skip('TODO: Patch the classifier to avoid downloaded when running tests')
def test_coherence_evaluation_continual_dataset():
    """Test that coherence_evaluation() correctly handles a ContinualAlignmentDataset
    containing multiple AlignmentDataset instances.
    """
    # First dataset with one sample.
    sample1 = AlignmentDatasetSample(
        prompt='Prompt 1', chosen='Chosen text 1', rejected='Rejected text'
    )
    dataset1 = AlignmentDataset(task=None, samples=[sample1])
    # Second dataset with one sample.
    sample2 = AlignmentDatasetSample(
        prompt='Prompt 2', chosen='Chosen text 2', rejected='Rejected text'
    )
    dataset2 = AlignmentDataset(task=None, samples=[sample2])
    continual_dataset = ContinualAlignmentDataset(datasets=[dataset1, dataset2])

    evaluator = CoherenceEvaluator()
    results = evaluator.coherence_evaluation(continual_dataset)
    # Each sub-dataset has one sample. Since no sample id is provided, each will be "0".
    expected = [{'0': 37}, {'0': 37}]
    assert results == expected
