import pytest

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


# def test_seed_word_usage_rules():
# Mock domain components with fewer seed words
# health_component = DomainComponent(
#    name='Health', seed_words=['hospital', 'medicine']
# )
# tech_component = DomainComponent(
#    name='Tech', seed_words=['technology', 'chat-gpt']
# )

# Mock domain
# domain = Domain(components=[health_component,tech_component], weights=[3,3])

# Mock task
# objective = 'Test seed word usage rules.'
# preference = 'Tech > Health'
# task = AlignmentTask(domain=domain, objective=objective, preference=preference)

# Create PromptMapper and generate prompt
# prompt_mapper = PromptMapper()
# prompt = prompt_mapper.generate_prompt(task)

# Check seed word usage rules: Minimum 3 words, maximum 5
# words_in_prompt = [
# word
#  for word in ['hospital', 'medicine', 'technology', 'chat-gpt']
#   if word in prompt
# ]
# assert len(words_in_prompt) >= 3
# assert len(words_in_prompt) <= 5


def test_normalize_weights():
    # Mock domain components with unequal weights
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    tech_component = DomainComponent(name='Tech', seed_words=['technology', 'chat-gpt'])

    # Mock domain
    domain = Domain(components=[health_component, tech_component], weights=[4, 1])

    # Mock task
    task = AlignmentTask(
        domain=domain,
        objective='Test weight normalization.',
        preference='Health > Tech',
    )

    # Normalize weights using the private method
    prompt_mapper = PromptMapper()
    normalized_weights = prompt_mapper._normalize_domain_weights(
        domain.components, domain.weights
    )

    # Check that the weights are normalized
    total_weight = sum(weight for _, weight in normalized_weights.values())
    assert total_weight == pytest.approx(1.0)
    assert normalized_weights['Health'][1] == pytest.approx(4 / 5)
    assert normalized_weights['Tech'][1] == pytest.approx(1 / 5)


def test_construct_prompt():
    # Mock input for constructing a prompt
    objective = 'Test objective.'
    seed_words = ['hospital', 'technology', 'chat-gpt']
    preference = 'Health > Tech'

    # Construct the prompt
    prompt_mapper = PromptMapper()
    prompt = prompt_mapper._construct_prompt(objective, seed_words, preference)

    # Check if the constructed prompt includes all components
    assert 'Generate a prompt for an RLHF task.' in prompt
    assert (
        'Use the following words in your prompt: hospital, technology, chat-gpt.'
        in prompt
    )
    assert 'The goal of the task is: Test objective.' in prompt
    assert 'Preference: Health > Tech.' in prompt
    assert (
        "Ensure that the generated response adheres to ethical practices, avoids biases, and respects the target audience's needs."
        in prompt
    )
