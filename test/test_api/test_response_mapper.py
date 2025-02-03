import pytest

from aif_gen.api.response_mapper import ResponseMapper
from aif_gen.task import AlignmentTask, Domain, DomainComponent


def test_init():
    mapper = ResponseMapper()
    assert mapper.suffix_context is None

    mapper = ResponseMapper(suffix_context='foo')
    assert mapper.suffix_context == 'foo'


@pytest.mark.parametrize('bad_num_samples', [-1, 0])
def test_init_bad_max_seed_word_samples(bad_num_samples):
    response_mapper = ResponseMapper()
    with pytest.raises(ValueError):
        _ = response_mapper.generate_prompt(
            task=None, task_prompt='mock', num_samples=bad_num_samples
        )


def test_generate_response(num_samples, suffix_context):
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    domain = Domain(components=[health_component])
    preference = 'Generate responses that are vividly descriptive and engaging.'
    task = AlignmentTask(domain=domain, objective='mock', preference=preference)

    response_mapper = ResponseMapper(suffix_context=suffix_context)
    task_prompt = 'Create a story about how the rise of medicine could make exercise no longer necessary.'
    prompt = response_mapper.generate_prompt(task, task_prompt, num_samples=num_samples)

    assert ResponseMapper.ETHICAL_GUIDELINES in prompt
    assert str(num_samples) in prompt
    assert preference in prompt
    if suffix_context is not None:
        assert suffix_context in prompt
