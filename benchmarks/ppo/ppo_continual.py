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
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side='left',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.value_model_path + '_0',
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )
    model = str(training_args.sft_model_path)
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
    base_output_dir = training_args.output_dir

    # Extract clean dataset name (e.g. "debug" from "benchmarks/continual_data_debug.json")
    clean_dataset_name = os.path.basename(script_args.dataset_name)
    if '.' in clean_dataset_name:
        clean_dataset_name = clean_dataset_name.split('.')[0]

    for i, dataset in enumerate(continual_dataset):
        # Build a custom repository name for the ppo model
        # e.g. "Qwen2-0.5B-Instruct_debug_PPO_0"
        custom_repo_name = (
            str(model).split('/')[-1] + '_' + clean_dataset_name + '_PPO_' + str(i)
        )
        if training_args.push_to_hub:
            training_args.hub_model_id = custom_repo_name
        # Update the output directory so that saving and pushing are done from a single folder.
        training_args.output_dir = os.path.join(base_output_dir, custom_repo_name)

        # Load the reward model following a similar naming convention.
        # Here we assume that the reward_model_path is expected to include the suffix.
        # e.g. "path/to/reward_model_Qwen2-0.5B-Instruct_debug_REWARD_0"
        reward_model_repo = (
            model.split('/')[-1] + '_' + clean_dataset_name + '_REWARD_' + str(i)
        )
        # Either load using training_args.reward_model_path with the suffix or adjust as needed.
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path + '_' + str(i), num_labels=1
        )

        dataset_text_field = 'prompt'
        dataset_train = dataset[script_args.dataset_train_split]
        dataset_test = dataset[script_args.dataset_test_split]

        def prepare_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
            def tokenize_batch(batch: dict) -> dict[str, Any]:
                # Get the texts from the expected field
                texts = batch.get(dataset_text_field)
                if texts is None:
                    # fallback to the first available column, if needed
                    first_key = list(batch.keys())[0]
                    texts = batch[first_key]
                outputs = tokenizer(
                    texts,
                    padding='max_length',  # adjust as needed
                    truncation=True,
                    max_length=2048,  # adjust max length as needed
                )
                return {'input_ids': outputs['input_ids']}

            return ds.map(
                tokenize_batch,
                batched=True,
                remove_columns=ds.column_names,  # optionally you can remove this for debugging
                # num_proc=training_args.dataset_num_proc  # remove num_proc for now
            )

        mapped_ds = prepare_dataset(dataset_train, tokenizer)
        for i in range(3):
            print(mapped_ds[i])

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
        print(trainer.dataloader)
        for batch in trainer.dataloader:
            print(batch)
            break
        trainer.train()

        if training_args.eval_strategy != 'no':
            metrics = trainer.evaluate()
            if i == 0:
                trainer.log({'dataset': {'name': script_args.dataset_name}})
            metrics['dataset'] = i
            print(f'eval/dataset/{i}')
            trainer.log_metrics(f'eval/dataset/{i}', metrics)
            trainer.save_metrics(f'eval', metrics)
            wandb_log({'eval': {'last': metrics}})
            wandb_log({f'task/{custom_repo_name}/last': metrics})

        # Save and push to hub.
        trainer.save_model(os.path.join(training_args.output_dir, 'last'))
        if training_args.push_to_hub:
            trainer.push_to_hub(
                model_name=custom_repo_name,
                dataset_name='Continual_PPO_' + clean_dataset_name + '_' + str(i),
            )

        # Clean up DeepSpeed engine if it exists.
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            del trainer.deepspeed
        torch.cuda.empty_cache()


if __name__ == '__main__':
    dataclass_types = (ContinualPPOArguments, ContinualPPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
