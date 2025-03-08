"""Adaptation of the PPO TRL training script for continual learning with task-based logging."""

import os

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

import wandb as wb
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

    ################
    # Model & Tokenizer Setup
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
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Load value model and policy model (main model)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.value_model_path,
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

    ################
    # Dataset Loading
    ################
    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name, mock=training_args.mock
    )
    base_output_dir = training_args.output_dir

    # Extract clean dataset name for repo naming
    clean_dataset_name = os.path.basename(script_args.dataset_name)
    if '.' in clean_dataset_name:
        clean_dataset_name = clean_dataset_name.split('.')[0]

    ################
    # Task Loop
    ################
    for i, dataset in enumerate(continual_dataset):
        # Build custom repository name for this task
        custom_repo_name = (
            model.split('/')[-1] + '_' + clean_dataset_name + '_PPO_' + str(i)
        )
        if training_args.push_to_hub:
            training_args.hub_model_id = custom_repo_name
        training_args.output_dir = os.path.join(base_output_dir, custom_repo_name)

        # Load reward model based on naming convention (expects suffix with task index)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path + '_' + str(i), num_labels=1
        )

        dataset_text_field = 'prompt'
        dataset_train = dataset[script_args.dataset_train_split]
        dataset_test = dataset[script_args.dataset_test_split]

        # Pre-tokenize the dataset to avoid issues during training
        def prepare_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
            def tokenize(element: dict) -> dict:
                outputs = tokenizer(
                    element[dataset_text_field],
                    padding=False,
                )
                return {'input_ids': outputs['input_ids']}

            return ds.map(
                tokenize,
                batched=True,
                remove_columns=ds.column_names,
                num_proc=training_args.dataset_num_proc,
            )

        with PartialState().local_main_process_first():
            train_dataset = prepare_dataset(dataset_train, tokenizer)
            eval_dataset = prepare_dataset(dataset_test, tokenizer)

        ################
        # Training and Evaluation
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
        # Set current task in trainer for task-based logging
        trainer.set_task(f'task_{i}')

        print(f'Training on task: {custom_repo_name}')
        trainer.train()

        if training_args.eval_strategy != 'no':
            # Mark final evaluation for this task so metrics are logged under eval.last as well
            trainer.mark_final_eval(True)
            metrics = trainer.evaluate()
            trainer.mark_final_eval(False)

            # Log dataset and task-specific metrics
            if i == 0:
                trainer.log({'dataset': {'name': script_args.dataset_name}})
            metrics['dataset'] = i
            print(f'Evaluation metrics for dataset {i}: {metrics}')
            trainer.log_metrics(f'eval/dataset/{i}', metrics)
            trainer.save_metrics('eval', metrics)

            # Log metrics to WandB
            wb.log({'eval': {'last': metrics}})  # type: ignore[attr-defined]
            wb.log({f'task/{custom_repo_name}/last': metrics})  # type: ignore[attr-defined]

        # Save model checkpoint and optionally push
        if not training_args.push_to_hub:
            trainer.save_model(os.path.join(training_args.output_dir))
        else:
            trainer.push_to_hub(
                model_name=custom_repo_name,
                dataset_name='Continual_PPO_' + clean_dataset_name + '_' + str(i),
            )

        # Cleanup DeepSpeed engine if used and free GPU memory
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            del trainer.deepspeed
        torch.cuda.empty_cache()

    print('Training completed for all tasks!')


if __name__ == '__main__':
    dataclass_types = (ContinualPPOArguments, ContinualPPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
