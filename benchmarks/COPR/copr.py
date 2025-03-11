import os
import random

import torch
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

from benchmarks.copr.copr_trainer import COPRArguments, COPRConfig, COPRTrainer
from benchmarks.dataloading import init_continual_dataset


def main(
    script_args: COPRArguments, training_args: COPRConfig, model_args: ModelConfig
) -> None:
    # Determine torch dtype and quantization configs
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

    # Load main model and (optionally) reference model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None

    # Load tokenizer and set chat template if needed
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Distributed training hack
    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Initialize continual dataset
    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name, mock=training_args.mock
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

    # Initialize the memory buffer at the script level
    memory_buffer: list = []

    for i, dataset in enumerate(continual_dataset):
        current_dataset_name = f'dataset-{i}'
        training_args.output_dir = f'{output_dir}/{current_dataset_name}'

        # Load reward model if path provided
        reward_model = None
        if training_args.reward_model_path is not None:
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(training_args.reward_model_path, str(i)), num_labels=1
            )

        # Process dataset
        eval_policy_dataset = dataset[script_args.dataset_test_split]
        dataset_train = dataset[script_args.dataset_train_split]
        dataset_test = dataset[script_args.dataset_test_split]

        # Convert to list for buffer management
        task_samples = list(dataset_train)
        combined_dataset = Dataset.from_list(memory_buffer + task_samples)

        if not combined_dataset:
            print(f'Warning: combined_dataset is empty for dataset-{i}.')
            continue

        # Initialize trainer with combined dataset
        trainer = COPRTrainer(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            args=training_args,
            train_dataset=combined_dataset,
            eval_dataset=dataset_test if training_args.eval_strategy != 'no' else None,
            eval_policy_dataset=eval_policy_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.set_task(f'task_{i}')

        # Train on combined dataset (current task + buffer)
        print(f'Training dataset: {current_dataset_name}')
        trainer.train()

        # Update memory buffer with samples from current task
        if getattr(training_args, 'use_buffer_ratio', False):
            buffer_size = max(1, int(len(task_samples) * training_args.buffer_ratio))
        else:
            buffer_size = training_args.memory_buffer_size

        # Add samples to buffer
        if len(task_samples) > buffer_size:
            samples_to_add = random.sample(task_samples, buffer_size)
        else:
            samples_to_add = task_samples

        # Update memory buffer
        memory_buffer.extend(samples_to_add)

        # Ensure buffer stays within size limit
        max_buffer_size = training_args.memory_buffer_size
        if len(memory_buffer) > max_buffer_size:
            memory_buffer = random.sample(memory_buffer, max_buffer_size)

        # Evaluate and log metrics to WandB
        if training_args.eval_strategy != 'no':
            # Mark this as the final evaluation for the current task
            trainer.mark_final_eval(True)

            # Run evaluation
            metrics = trainer.evaluate()

            # Reset final eval flag
            trainer.mark_final_eval(False)

            # Save metrics to file
            if i == 0:
                trainer.log({'dataset_name': script_args.dataset_name})

            # Include dataset index in metrics
            metrics['dataset'] = i
            trainer.save_metrics('eval', metrics)

        # Save model checkpoint for the task
        trainer.save_model(os.path.join(training_args.output_dir, 'last'))
        if training_args.push_to_hub:
            trainer.push_to_hub(
                dataset_name=f'Continual_COPR_{script_args.dataset_name}/dataset-{i}'
            )

        # If using DeepSpeed, cleanup the engine and free GPU memory
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            del trainer.deepspeed
        torch.cuda.empty_cache()

    print('Training completed for all tasks!')


if __name__ == '__main__':
    dataclass_types = (COPRArguments, COPRConfig, ModelConfig)
    parser = TrlParser(dataclass_types)
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
