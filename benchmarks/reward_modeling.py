import warnings
from dataclasses import dataclass, field
from typing import Dict

import submitit
import torch
from dataloading import init_continual_dataset
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)


@dataclass
class ExtendedScriptArguments(ScriptArguments):
    dataset_index: int = field(
        default=0,
        metadata={
            'help': 'Index of the dataset to use, dataset points to ContinualDataset, '
            'this index points to individual dataset in the ContinualDataset.'
        },
    )
    all_datasets: bool = field(
        default=False,
        metadata={'help': 'Whether to use all datasets in the ContinualDataset.'},
    )
    slurm_partition: str = field(
        default='main',
        metadata={'help': 'Slurm partition to use.'},
    )
    slurm_timeout_min: int = field(
        default=60,
        metadata={'help': 'Slurm job timeout in minutes.'},
    )
    slurm_gpus: int = field(
        default=1,
        metadata={'help': 'GPUs to use per job.'},
    )
    slurm_cpus_per_task: int = field(
        default=4,
        metadata={'help': 'Number of CPUs per Slurm task.'},
    )
    slurm_mem_gb: int = field(
        default=24,
        metadata={'help': 'Memory required per Slurm task in GB.'},
    )
    slurm_constraint: str = field(
        default='volta',
        metadata={'help': 'Slurm constraint to use.'},
    )


def train_model(
    script_args: ExtendedScriptArguments,
    training_args: RewardConfig,
    model_args: ModelConfig,
    dataset: Dict[str, Dataset],
    index: int,
) -> None:
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # The code is heavely based on the reward_modeling script from the trl library, the only difference is we handle dataset_index as an argument

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.use_peft and model_args.lora_task_type != 'SEQ_CLS':
        warnings.warn(
            'You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs'
            ' Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.',
            UserWarning,
        )

    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != 'no'
        else None,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    print(f'Saving model {index} to: {training_args.output_dir}')
    trainer.save_model(
        training_args.output_dir + f'/{script_args.dataset_name}/' + str(index)
    )

    if training_args.eval_strategy != 'no':
        metrics = trainer.evaluate()
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name + str(index))


if __name__ == '__main__':
    parser = HfArgumentParser((ExtendedScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name
    )

    if script_args.all_datasets:
        executor = submitit.AutoExecutor(folder='submitit_logs')
        executor.update_parameters(
            timeout_min=script_args.slurm_timeout_min,
            slurm_partition=script_args.slurm_partition,
            gpus_per_node=script_args.slurm_gpus,
            cpus_per_task=script_args.slurm_cpus_per_task,
            mem_gb=script_args.slurm_mem_gb,
            constraint=script_args.slurm_constraint,
        )

        print(f'Submitting {len(continual_dataset)} training jobs...')

        jobs = []
        for index, dataset in enumerate(continual_dataset):
            print(f'Submitting job {index + 1}/{len(continual_dataset)}')
            job = executor.submit(
                train_model, script_args, training_args, model_args, dataset, index
            )
            jobs.append(job)

        print('Waiting for jobs to complete...')
        for i, job in enumerate(jobs):
            print(f'Waiting for job {i + 1}/{len(jobs)}')
            try:
                job.result()
            except Exception as e:
                print(f'Job {i + 1} failed with error: {e}')
    else:
        dataset = continual_dataset[script_args.dataset_index]
        train_model(
            script_args, training_args, model_args, dataset, script_args.dataset_index
        )
