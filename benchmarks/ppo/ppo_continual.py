"""Adaption of the PPO TRL training script for continual learning."""

import os
from typing import Any

import torch
from accelerate import PartialState
from continual_ppo_trainer import (
    ContinualPPOArguments,
    ContinualPPOConfig,
    ContinualPPOTrainer,
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
from wandb import log as wandb_log  # type: ignore


def main(
    script_args: ContinualPPOArguments,
    training_args: ContinualPPOConfig,
    model_args: ModelConfig,
) -> None:
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.value_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name, mock=training_args.mock
    )
    output_dir = training_args.output_dir

    for i, _ in enumerate(continual_dataset):
        reward_path = os.path.join(training_args.reward_model_path, str(i))
        if not os.path.exists(reward_path):
            raise FileNotFoundError(
                f'Reward model not found for dataset {i} at {reward_path}'
            )

    for i, dataset in enumerate(continual_dataset):
        current_dataset_name: str = f'dataset-{i}'
        training_args.output_dir = f'{output_dir}/{current_dataset_name}'

        # TODO Value model is based on the reward model, so we need to think if we want to reinstantiate
        #  the value model with the new reward model when we change dataset or continue to train with the old one
        # Reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path + f'/{str(i)}', num_labels=1
        )

        dataset_text_field = 'prompt'
        dataset_train = dataset[script_args.dataset_train_split]
        dataset_test = dataset[script_args.dataset_test_split]

        def prepare_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
            """pre-tokenize the dataset before training; only collate during training."""

            def tokenize(element: dict) -> dict[str, Any]:
                if not training_args.mock:
                    # explicit dataset
                    outputs = tokenizer(
                        element[dataset_text_field],
                        padding=False,
                    )
                else:
                    # implicit dataset type
                    outputs = tokenizer(
                        element,
                        padding=False,
                    )
                return {'input_ids': outputs['input_ids']}

            return dataset.map(
                tokenize,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=training_args.dataset_num_proc,
            )

        with PartialState().local_main_process_first():
            train_dataset = prepare_dataset(dataset_train, tokenizer)
            eval_dataset = prepare_dataset(dataset_test, tokenizer)

        ################
        # Training
        ################
        trainer = ContinualPPOTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )
        trainer.train()

        # TODO: PPOTrainer doens't have an evaluate method - should be fixed now
        if training_args.eval_strategy != 'no':
            metrics = trainer.evaluate()
            if i == 0:
                trainer.log({'dataset': {'name': script_args.dataset_name}})
            metrics['dataset'] = i
            # Log evaluation metrics under a hierarchy using slashes for wandb
            print(f'eval/dataset/{i}')
            trainer.log_metrics(f'eval/dataset/{i}', metrics)
            trainer.save_metrics(f'eval', metrics)
            # ToDo: we can't use trainer.log here because it repeats computations of some the metrics, that can be heavy
            wandb_log({'eval': {'last': metrics}})
            wandb_log({f'task/{current_dataset_name}/last': metrics})

        # Save and push to hub
        trainer.save_model(training_args.output_dir + '/last')
        if training_args.push_to_hub:
            trainer.push_to_hub(
                dataset_name=(
                    'Continual_PPO_' + script_args.dataset_name + f'/dataset-{i}'
                )
            )

        # If using DeepSpeed through Accelerate, tear down the engine after training.
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            # Remove reference to the DeepSpeed engine to allow proper cleanup.
            del trainer.deepspeed
        # Free cached GPU memory.
        torch.cuda.empty_cache()


if __name__ == '__main__':
    dataclass_types = (ContinualPPOArguments, ContinualPPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)

    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
