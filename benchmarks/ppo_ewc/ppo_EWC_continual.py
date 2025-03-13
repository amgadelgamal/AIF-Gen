"""Adaptation of the PPO training script for continual learning with EWC regularization."""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    AutoModelForCausalLMWithValueHead,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import wandb as wb
from benchmarks.dataloading import init_continual_dataset
from benchmarks.ppo_ewc.continual_ppo_EWC_trainer import (
    ContinualPPOEWCArguments,
    ContinualPPOEWCConfig,
    ContinualPPOEWCTrainer,
)


def main(
    script_args: ContinualPPOEWCArguments,
    training_args: ContinualPPOEWCConfig,
    model_args: ModelConfig,
) -> None:
    """Main entry point for PPO training with EWC."""
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )
    if script_args.wandb_run_name is not None:
        training_args.run_name = script_args.wandb_run_name

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    # Add value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # Configure PEFT if needed
    peft_config = get_peft_config(model_args)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DDPT distributed setup
    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load continual datasets
    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name, mock=training_args.mock
    )
    output_dir = training_args.output_dir

    # Process each dataset in the continual learning setup
    for i, dataset in enumerate(continual_dataset):
        current_dataset_name: str = f'dataset-{i}'
        training_args.output_dir = f'{output_dir}/{current_dataset_name}'

        # Load reward model for this task (if available)
        reward_model = None
        if script_args.value_model_path:
            reward_model = AutoModelForCausalLM.from_pretrained(
                f'{script_args.value_model_path}_{str(i)}'
            )

        # Prepare datasets
        dataset_train = dataset['train']
        dataset_test = dataset.get('test', None)

        # Initialize trainer
        trainer = ContinualPPOEWCTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=model,
            reward_model=reward_model,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            peft_config=peft_config,
        )

        # Set task name for metric tracking
        trainer.set_task(f'task-{i}')

        print('Training dataset:', current_dataset_name)
        trainer.train()

        # Evaluate if dataset_test is available
        if dataset_test is not None:
            trainer.mark_final_eval(True)
            metrics = trainer.evaluate()

            # Log dataset info on first task
            if i == 0:
                trainer.log({'dataset_name': script_args.dataset_name})

            # Log metrics with task identifier
            metrics['dataset'] = i
            trainer.log_metrics(f'eval/dataset/{i}', metrics)
            trainer.save_metrics(f'eval', metrics)
            wb.log({'eval': {'last': metrics}})  # type: ignore[attr-defined]
            wb.log({f'task/{current_dataset_name}/last': metrics})  # type: ignore[attr-defined]

        # Save model
        trainer.save_model(training_args.output_dir + '/last')
        if training_args.push_to_hub:
            trainer.push_to_hub(
                dataset_name=(
                    'Continual_PPO_EWC_' + script_args.dataset_name + f'/dataset-{i}'
                )
            )

        # Clean up for next task
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            # Remove reference to the DeepSpeed engine to allow proper cleanup.
            del trainer.deepspeed
        # Free cached GPU memory
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # Parse arguments
    dataclass_types = (ContinualPPOEWCArguments, ContinualPPOEWCConfig, ModelConfig)
    parser = TrlParser(dataclass_types)

    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
