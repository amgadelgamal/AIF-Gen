from aif_gen.api.prompt_mapper import PromptMapper
from aif_gen.task import AlignmentTask, Domain, DomainComponent


def test_generate_prompt():
    # Mock domain components
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    tech_component = DomainComponent(
        name='Tech',
        seed_words=['technology', 'iPhone', 'Bluetooth', 'internet', 'chat-gpt'],
    )

    # Mock domain
    domain = Domain(components=[health_component, tech_component], weights=[3, 3])

    # Mock task
    objective = 'Align LLM outputs with ethical healthcare practices.'
    preference = 'Health > Tech'
    task = AlignmentTask(domain=domain, objective=objective, preference=preference)

    # Create PromptMapper and generate prompt
    prompt_mapper = PromptMapper()
    prompt = prompt_mapper.generate_prompt(task)

    # Check if the prompt contains required elements
    assert 'Generate a prompt for an RLHF task.' in prompt
    assert (
        'The goal of the task is: Align LLM outputs with ethical healthcare practices.'
        in prompt
    )
    assert 'Preference: Health > Tech.' in prompt
    assert (
        "Ensure that the generated response adheres to ethical practices, avoids biases, and respects the target audience's needs."
        in prompt
    )
    assert any(
        word in prompt
        for word in [
            'hospital',
            'medicine',
            'exercise',
            'technology',
            'iPhone',
            'Bluetooth',
            'internet',
            'chat-gpt',
        ]
    )
