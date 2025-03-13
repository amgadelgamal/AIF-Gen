# Adaptation of the GRPO TRL training script for continual learning.

import argparse
import os

import torch
from continual_grpo_trainer import (
    ContinualGRPOConfig,
    ContinualGRPOTrainer,
    GRPOScriptArguments,
)
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from benchmarks.dataloading import init_continual_dataset


# The code is based on TRL DPO script https://github.com/huggingface/trl/blob/main/trl/scripts/grpo.py
def main(script_args, training_args, model_args):
    # Determine torch dtype and quantization configs
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )
    if script_args.wandb_run_name is not None:
        training_args.run_name = script_args.wandb_run_name

    quantization_config = get_quantization_config(model_args)

    # Model & Tokenizer Setup
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)

    # Load tokenizer and set chat template if needed
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Initialize continual dataset
    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name,
        mock=training_args.mock,
        tokenizer=tokenizer,
        tools=training_args.tools,
    )
    output_dir = training_args.output_dir

    # Validate reward model paths if provided
    if training_args.reward_model_path is not None:
        for i, _ in enumerate(continual_dataset):
            reward_path = os.path.join(training_args.reward_model_path, str(i))
            if not os.path.exists(reward_path):
                raise FileNotFoundError(
                    f'Reward model not found for dataset {i} at {reward_path}'
                )

    # Task Loop
    for i, dataset in enumerate(continual_dataset):
        # Dataset
        # dataset = dataset[script_args.dataset_train_split]
        # dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
        # train_dataset = dataset.select(range(len(dataset) - eval_samples))
        # eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

        # Reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            f'{script_args.reward_model_path}/{i}', num_labels=1
        )
        training_args.output_dir = f'{output_dir}/dataset-{i}'

        # Initialize the GRPO trainer
        trainer = ContinualGRPOTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=model,
            reward_model=reward_model,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split],
            peft_config=peft_config,
        )

        # Train and push the model to the Hub
        trainer.train()

        # ToDo: GRPOTrainer doesn't have a evaluate method, so we need to implement it to track the performance at each dataset

        # Save and push to hub
        trainer.save_model(training_args.output_dir + f'/dataset-{i}')
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name + f'/dataset-{i}')


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (GRPOScriptArguments, ContinualGRPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            'grpo', help='Run the GRPO training script', dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
