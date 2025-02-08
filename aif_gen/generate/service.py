import asyncio
import logging
from typing import AsyncGenerator, List

import backoff
import openai
import pydantic
from tqdm.asyncio import tqdm

from aif_gen.api.prompt_mapper import PromptMapper
from aif_gen.api.response_mapper import ResponseMapper
from aif_gen.dataset import AlignmentDataset, AlignmentDatasetSample
from aif_gen.task.alignment_task import AlignmentTask

try:
    client = openai.AsyncOpenAI()
except (openai.OpenAIError, Exception) as e:
    logging.exception(e)


async def process_tasks(
    tasks: List[AlignmentTask],
    num_samples: int,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> AsyncGenerator[AlignmentDataset, None]:
    coros = [
        generate_dataset(task, num_samples, model_name, async_semaphore)
        for task in tasks
    ]
    for coro in tqdm(asyncio.as_completed(coros), total=len(coros)):
        dataset = await coro
        yield dataset


async def generate_dataset(
    task: AlignmentTask,
    num_samples: int,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> AlignmentDataset:
    task_prompt = await generate_task_prompt(task, model_name, async_semaphore)
    samples = await generate_samples(
        task, task_prompt, num_samples, model_name, async_semaphore
    )
    return AlignmentDataset(task, samples)


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def generate_task_prompt(
    task: AlignmentTask,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> str:
    logging.info(f'Generating task prompt for {task}')
    prompt_mapper = PromptMapper()
    prompt = prompt_mapper.generate_prompt(task)
    logging.debug(f'Using task prompt: {prompt}')

    async with async_semaphore:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
        )
    output = response.choices[0].message.content
    assert output is not None
    logging.info(f'Generated task prompt: {output}')
    return output


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def generate_samples(
    task: AlignmentTask,
    task_prompt: str,
    num_samples: int,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
) -> List[AlignmentDatasetSample]:
    logging.info(f'Generating {num_samples} response pairs for {task}')
    response_mapper = ResponseMapper()
    prompt = response_mapper.generate_prompt(task, task_prompt, num_samples)
    logging.debug(f'Using response prompt: {prompt}')

    class _ResponsePair(pydantic.BaseModel):
        chosen: str
        rejected: str

    class _ResponsePairList(pydantic.BaseModel):
        responses: List[_ResponsePair]

    async with async_semaphore:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'SyntheticPreference',
                    'schema': _ResponsePairList.model_json_schema(),
                    'strict': True,
                },
            },
        )
    output = response.choices[0].message.content
    assert output is not None

    try:
        responses = _ResponsePairList.model_validate_json(output).responses
        logging.debug(f'Received {len(responses)} response pairs: {responses}')
        if len(responses) != num_samples:
            logging.warning(
                f'Requested {num_samples} response pairs LM generated {len(responses)}'
            )
        return [
            AlignmentDatasetSample(
                prompt=task_prompt, chosen=response.chosen, rejected=response.rejected
            )
            for response in responses
        ]
    except pydantic.ValidationError as e:
        logging.exception(e)
        return []
