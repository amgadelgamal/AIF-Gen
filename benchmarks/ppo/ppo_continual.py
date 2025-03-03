"""Adaption of the PPO TRL training script for continual learning."""

import os

import torch
import wandb
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
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    value_model_path = (
        '/home/mila/i/ivan.anokhin/AIF-Gen/Qwen/Qwen2-0.5B-Reward/debug/0'
    )

    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

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

        # ToDo: Value model is based on the reward model, so we need to think if we want to reinstantiate
        #  the value model with the new reward model when we change dataset or continue to train with the old one
        # Reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path + f'/{str(i)}', num_labels=1
        )

        if not training_args.mock:

            def concat_prompt_to_completions(example: dict) -> dict[str, list[int]]:
                return {
                    'chosen': example['prompt'] + example['chosen'],
                    'rejected': example['prompt'] + example['rejected'],
                }

            dataset_train = dataset[script_args.dataset_train_split].map(
                concat_prompt_to_completions, remove_columns='prompt'
            )
            dataset_test = dataset[script_args.dataset_test_split].map(
                concat_prompt_to_completions, remove_columns='prompt'
            )
        else:
            dataset_train = dataset[script_args.dataset_train_split]
            dataset_test = dataset[script_args.dataset_test_split]

        trainer = ContinualPPOTrainer(
            args=training_args,
            model=policy,
            ref_model=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        trainer.train()

        # TODO: PPOTrainer doens't have an evaluate method
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
            wandb.log({'eval': {'last': metrics}})
            wandb.log({f'task/{current_dataset_name}/last': metrics})

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
