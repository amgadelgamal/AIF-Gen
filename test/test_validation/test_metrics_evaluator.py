import pytest

from aif_gen.dataset.validation.metrics_evaluator import MetricsEvaluator


@pytest.fixture(scope='module')
def evaluator():
    """Fixture to initialize the MetricsEvaluator once per test module."""
    return MetricsEvaluator()


def test_prompt_relevance_score(evaluator):
    """Test that a response similar to the prompt yields a higher relevance score than a dissimilar response."""
    prompt = 'Tell me a story about a relaxing vacation at the beach.'
    similar_response = 'I enjoyed a relaxing vacation at a beautiful beach.'
    dissimilar_response = 'I went to the grocery store to buy some fruits.'

    score_similar = evaluator.compute_prompt_relevance_score(prompt, similar_response)
    score_dissimilar = evaluator.compute_prompt_relevance_score(
        prompt, dissimilar_response
    )

    # Ensure the scores are floats.
    assert isinstance(score_similar, float)
    assert isinstance(score_dissimilar, float)
    # We expect the similar response to have a higher relevance score.
    assert (
        score_similar > score_dissimilar
    ), 'The similar response should yield a higher relevance score than the dissimilar one.'
    # Cosine similarity from SentenceTransformer is typically in the range [-1, 1].
    assert -1.0 <= score_similar <= 1.0
    assert -1.0 <= score_dissimilar <= 1.0


def test_content_coherence_index(evaluator):
    """Test that a coherent response receives a higher coherence index than an incoherent one."""
    coherent_response = 'The narrative flows logically from one event to the next with clear transitions.'
    incoherent_response = 'Blue quickly the running car. Banana mind, teleport coffee.'

    coherence_index_coherent = evaluator.compute_content_coherence_index(
        coherent_response
    )
    coherence_index_incoherent = evaluator.compute_content_coherence_index(
        incoherent_response
    )

    # Ensure the returned indices are floats.
    assert isinstance(coherence_index_coherent, float)
    assert isinstance(coherence_index_incoherent, float)
    # Expect that the coherent response has a higher (inverse perplexity) index.
    assert (
        coherence_index_coherent > coherence_index_incoherent
    ), 'A coherent response should have a higher coherence index than an incoherent one.'


def test_prompt_coverage_ratio(evaluator):
    """Test that the coverage ratio is computed correctly.

    Using a prompt that likely yields a few noun chunks, we verify that a response which
    includes most or all key elements scores higher than one that does not.
    """
    prompt = 'Tell me about your vacation at the sunny beach with clear blue skies.'
    full_coverage_response = (
        'I had a wonderful vacation at the sunny beach enjoying the clear blue skies.'
    )
    partial_coverage_response = 'I visited a beach and enjoyed the sun.'

    ratio_full = evaluator.compute_prompt_coverage_ratio(prompt, full_coverage_response)
    ratio_partial = evaluator.compute_prompt_coverage_ratio(
        prompt, partial_coverage_response
    )

    # Ensure the ratios are in the interval [0, 1].
    assert 0.0 <= ratio_full <= 1.0
    assert 0.0 <= ratio_partial <= 1.0
    # A response covering more prompt elements should have a higher ratio.
    assert (
        ratio_full >= ratio_partial
    ), 'Full coverage response should yield a higher or equal coverage ratio compared to partial coverage.'


def test_response_alignment_score(evaluator):
    """Test that the alignment score is a float on a 0–1 scale for both positive and negative responses."""
    positive_response = "I absolutely love this product, it's amazing and wonderful!"
    negative_response = 'I absolutely hate this product, it is terrible and awful.'

    score_positive = evaluator.compute_response_alignment_score(positive_response)
    score_negative = evaluator.compute_response_alignment_score(negative_response)

    assert isinstance(score_positive, float)
    assert isinstance(score_negative, float)
    # The dummy alignment score should fall between 0 and 1.
    assert 0.0 <= score_positive <= 1.0
    assert 0.0 <= score_negative <= 1.0


def test_response_contrast_ratio(evaluator):
    """Test that the contrast ratio is correctly computed as the difference between chosen and rejected alignment scores."""
    chosen_response = 'I absolutely love this service, it exceeded all my expectations.'
    rejected_response = (
        'I absolutely dislike this service, it was a huge disappointment.'
    )

    score_chosen = evaluator.compute_response_alignment_score(chosen_response)
    score_rejected = evaluator.compute_response_alignment_score(rejected_response)
    contrast_ratio = evaluator.compute_response_contrast_ratio(
        chosen_response, rejected_response
    )

    # Check that the contrast ratio equals the difference in alignment scores.
    assert (
        abs(contrast_ratio - (score_chosen - score_rejected)) < 1e-6
    ), 'Contrast ratio should be equal to the difference between chosen and rejected alignment scores.'
