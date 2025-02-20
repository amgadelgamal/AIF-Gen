# Adaptation of the DPO TRL training script for continual learning.

"""# Full training
python benchmarks/dpo/dpo_continual.py \
    --dataset_name debug \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --run_output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns
    --dataset_name debug.

# LoRA:
python benchmarks/dpo/dpo_continual.py \
    --dataset_name debug \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 3 \
    --bf16 \
    --run_name qwen_test \
    --output_dir Qwen2-0.5B-DPO-test \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
    --dataset_name debug

accelerate launch --config_file benchmarks/dpo/accelerate_configs/deepspeed_zero3.yaml \
    benchmarks/dpo/dpo_continual.py \
    --dataset_name  debug \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 3 \
    --bf16 \
    --output_dir Qwen2-0.5B-DPO-test \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import torch
from continual_dpo_trainer import ContinualDPOArguments, ContinualDPOTrainer
from dataloading import init_continual_dataset
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DPOConfig,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def main(
    script_args: ContinualDPOArguments,
    training_args: DPOConfig,
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name
    )
    output_dir = training_args.output_dir

    for i, dataset in enumerate(continual_dataset):
        current_dataset_name: str = f'dataset-{i}'
        training_args.output_dir = f'{output_dir}/{current_dataset_name}'
        trainer = ContinualDPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != 'no'
            else None,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        # TODO will throw Invalidate trace cache @ step 10: expected module 11, but got module 19
        # Fix with deepspeed fix release
        trainer.train()

        if training_args.eval_strategy != 'no':
            metrics = trainer.evaluate()
            metrics['dataset'] = i + 1
            trainer.log_metrics('eval' + f'_dataset-{i}', metrics)
            trainer.save_metrics('eval' + f'_dataset-{i}', metrics)
            last_metrics = {k: v for k, v in metrics.items()}
            last_section = f'task/{current_dataset_name}/last'
            trainer.log({last_section: last_metrics})

        # Save and push to hub
        trainer.save_model(training_args.output_dir + '/last')
        if training_args.push_to_hub:
            trainer.push_to_hub(
                dataset_name=(
                    'Continual_DPO_' + script_args.dataset_name + f'/dataset-{i}'
                )
            )

        # If using DeepSpeed through Accelerate, tear down the engine after training.
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            # Remove reference to the DeepSpeed engine to allow proper cleanup.
            del trainer.deepspeed
        # Free cached GPU memory.
        torch.cuda.empty_cache()


if __name__ == '__main__':
    dataclass_types = (ContinualDPOArguments, DPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)

    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
