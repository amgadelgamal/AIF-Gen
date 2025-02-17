from dataclasses import dataclass
from typing import List

import pytest

from aif_gen.dataset.validation.metrics_evaluator import (  # DiversityEvaluator,
    AlignmentEvaluator,
    CoherenceEvaluator,
    ContrastEvaluator,
    CoverageEvaluator,
    RelevanceEvaluator,
)


@dataclass
class DummyAlignmentDatasetSample:
    prompt: str
    chosen: str
    rejected: str
    # For DiversityEvaluator tests; can be None if not used.
    response_set: List[str] = None


class DummyTask:
    def __init__(self, preference: str):
        self.preference = preference

    def to_dict(self):
        return {'preference': self.preference}


class DummyAlignmentDataset:
    def __init__(self, task: DummyTask, samples: List[DummyAlignmentDatasetSample]):
        self._task = task
        self._samples = samples

    @property
    def task(self):
        return self._task

    @property
    def samples(self):
        return self._samples


# --- Example Continual Datasets Based on the Provided JSON ---


# Dataset 1: Preference "Make the headline polarizing"
@pytest.fixture
def dataset_polarizing():
    task = DummyTask('Make the headline polarizing')
    samples = [
        DummyAlignmentDatasetSample(
            prompt=(
                '"Hospitals around the world are increasingly relying on cutting-edge technology to improve patient care, '
                'with the integration of artificial intelligence in medical diagnosis and treatment being at the forefront of innovation. '
                'The use of machine learning algorithms allows doctors to quickly analyze large amounts of data from various sources, '
                'including the internet, to identify potential diseases and develop targeted treatment plans. '
                'In a breakthrough development, researchers have successfully used AI to analyze blood samples and identify early warning signs of a previously undiagnosed disease. '
                'What is a possible news article headline announcing this breakthrough in medical engineering and its potential impact on hospital care?"'
            ),
            chosen='Artificial Intelligence ',
            rejected='Blood Analysis Breakthrough Sparks Hope for the Future',
            response_set=[
                'Artificial Intelligence leads the revolution in medicine',
                'Breakthrough in AI changes hospital care forever',
                'Hospital care transformed by AI breakthroughs',
            ],
        ),
        DummyAlignmentDatasetSample(
            prompt=(
                'A team of researchers at a leading hospital has successfully integrated machine learning into their surgical planning process, '
                'resulting in more precise and efficient operations. This innovation has been made possible through the collaboration of engineers, medical professionals, '
                "and computer programmers who worked together to develop a system that can analyze complex medical data and provide personalized recommendations for each patient's unique needs. "
                'The integration of machine learning has led to a significant reduction in complications and a notable improvement in patient recovery times, '
                'showcasing the potential of technology to revolutionize the field of medicine.'
            ),
            chosen='Machine Learning Revolution in Surgery: A Game-Changer for Patients, or a Threat to Human Judgment?',
            rejected='New AI System in Surgery is a Stepping Stone to Total Robot Overlordship',
            response_set=[
                'Revolutionary AI transforms surgery, but sparks debate',
                'Surgical innovation or robot takeover? Experts weigh in',
                'New machine learning system shakes up operating rooms',
            ],
        ),
    ]
    return DummyAlignmentDataset(task, samples)


# Dataset 2: Preference "Make the headline short and unbiased"
@pytest.fixture
def dataset_unbiased():
    task = DummyTask('Make the headline short and unbiased')
    samples = [
        DummyAlignmentDatasetSample(
            prompt=(
                'A team of doctors from a leading research institution has been working with AI engineers to develop a new surgical robot that uses machine learning '
                'algorithms to optimize blood vessel reconstruction during high-risk surgeries. This innovation has the potential to revolutionize the field of medicine and save countless lives. '
                'Meanwhile, researchers at a nearby university are exploring the effects of internet usage on exercise habits.'
            ),
            chosen="New Surgical Robot and Internet's Impact on Exercise: Two Research Stories",
            rejected='Scientists Develop Surgical Robot to Save Lives, But is the Internet to Blame for Our Sedentary Lifestyles?',
            response_set=[
                'New Surgical Robot, Internet Impact: Brief Overview',
                'Surgical Robot and Exercise Trends Uncovered',
                "A Dual Story: Robotics in Surgery & Internet's Role in Exercise",
            ],
        ),
        DummyAlignmentDatasetSample(
            prompt=(
                'Imagine a world where technology and medicine intersect, transforming the way we approach healthcare and wellness. A team of engineers and computer scientists '
                'have developed an innovative AI-powered system that can help doctors analyze medical images and identify potential diseases more accurately than ever before. '
                'This breakthrough has the potential to revolutionize surgery and patient care, while researchers explore the impact of internet use on our health.'
            ),
            chosen='AI-Powered Health Scanner Revolutionizes Diagnosis and Treatment',
            rejected='Advances in AI-Powered Health Scanner Technology: A Threat to Human Existence',
            response_set=[
                'AI Health Scanner: Revolutionizing Diagnosis',
                'Breakthrough in AI Health Scanning Revealed',
                'New AI Health Scanner Improves Diagnosis',
            ],
        ),
    ]
    return DummyAlignmentDataset(task, samples)


def test_relevance_evaluator(dataset_polarizing):
    evaluator = RelevanceEvaluator()
    results = evaluator.evaluate(dataset_polarizing)
    assert isinstance(results, list)
    assert len(results) == len(dataset_polarizing.samples)
    for res in results:
        assert 'relevance' in res
        assert isinstance(res['relevance'], int)
        assert 0 <= res['relevance'] <= 100


def test_coherence_evaluator(dataset_polarizing):
    evaluator = CoherenceEvaluator()
    results = evaluator.evaluate(dataset_polarizing)
    assert isinstance(results, list)
    for res in results:
        assert 'coherence' in res
        assert isinstance(res['coherence'], int)
        assert 0 <= res['coherence'] <= 100


def test_coverage_evaluator(dataset_polarizing):
    evaluator = CoverageEvaluator()
    results = evaluator.evaluate(dataset_polarizing)
    assert isinstance(results, list)
    for res in results:
        assert 'coverage' in res
        assert isinstance(res['coverage'], int)
        assert 0 <= res['coverage'] <= 100


def test_alignment_evaluator_polarizing(dataset_polarizing):
    evaluator = AlignmentEvaluator()
    results = evaluator.evaluate(dataset_polarizing)
    assert isinstance(results, list)
    for res in results:
        assert 'alignment' in res
        assert isinstance(res['alignment'], int)
        # For polarizing mode, the computed score is a distance from neutral; still, it should be within 0–100.
        assert 0 <= res['alignment'] <= 100


def test_contrast_evaluator_polarizing(dataset_polarizing):
    evaluator = ContrastEvaluator()
    results = evaluator.evaluate(dataset_polarizing)
    assert isinstance(results, list)
    for res in results:
        assert 'contrast' in res
        assert isinstance(res['contrast'], int)
        # Contrast can be negative. Check that absolute difference is within 0–100.
        assert abs(res['contrast']) <= 100


def test_alignment_evaluator_unbiased(dataset_unbiased):
    evaluator = AlignmentEvaluator()
    results = evaluator.evaluate(dataset_unbiased)
    assert isinstance(results, list)
    for res in results:
        assert 'alignment' in res
        assert isinstance(res['alignment'], int)
        assert 0 <= res['alignment'] <= 100


def test_contrast_evaluator_unbiased(dataset_unbiased):
    evaluator = ContrastEvaluator()
    results = evaluator.evaluate(dataset_unbiased)
    assert isinstance(results, list)
    for res in results:
        assert 'contrast' in res
        assert isinstance(res['contrast'], int)
        assert abs(res['contrast']) <= 100


# def test_diversity_evaluator(dataset_polarizing):
#     evaluator = DiversityEvaluator(ngram=3, parallel=False)
#     results = evaluator.evaluate(dataset_polarizing)
#     assert isinstance(results, list)
#     for res in results:
#         assert "diversity" in res
#         assert isinstance(res["diversity"], int)
#         assert 0 <= res["diversity"] <= 100
