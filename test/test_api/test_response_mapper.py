import random

from aif_gen.api.response_mapper import ResponseMapper
from aif_gen.task import AlignmentTask, Domain, DomainComponent


def test_init():
    mapper = ResponseMapper()
    assert mapper.suffix_context is None

    mapper = ResponseMapper(suffix_context='foo')
    assert mapper.suffix_context == 'foo'


def test_generate_response(suffix_context):
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    domain = Domain(components=[health_component])
    preference = 'Generate responses that are vividly descriptive and engaging.'
    task = AlignmentTask(domain=domain, objective='mock', preference=preference)

    response_mapper = ResponseMapper(suffix_context=suffix_context)
    task_prompt = 'Create a story about how the rise of medicine could make exercise no longer necessary.'
    prompt = response_mapper.generate_prompt(task, task_prompt)

    assert ResponseMapper.ETHICAL_GUIDELINES in prompt
    assert preference in prompt
    if suffix_context is not None:
        assert suffix_context in prompt


def test_generate_no_preference_response(suffix_context):
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    domain = Domain(components=[health_component])
    preference = 'Generate responses that are vividly descriptive and engaging.'
    task = AlignmentTask(domain=domain, objective='mock', preference=preference)

    response_mapper = ResponseMapper(suffix_context=suffix_context)
    task_prompt = 'Create a story about how the rise of medicine could make exercise no longer necessary.'
    # generate a list of random scores each between 1 and 5
    scores = [random.randint(1, 5) for _ in range(len(response_mapper.preference_axes))]
    prompt = response_mapper.generate_no_preference_prompt(task, task_prompt, scores)

    assert ResponseMapper.ETHICAL_GUIDELINES in prompt
    assert preference not in prompt
    for pref1, pref2 in response_mapper.preference_axes:
        assert pref1 in prompt
        assert pref2 in prompt
    # assert scores
    for i, (pref1, pref2) in enumerate(response_mapper.preference_axes):
        assert str(scores[i]) in prompt
    if suffix_context is not None:
        assert suffix_context in prompt
