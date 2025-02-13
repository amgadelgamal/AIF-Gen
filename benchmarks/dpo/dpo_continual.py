# Adaptation of the DPO TRL training script for continual learning.

"""
# Full training
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

# LoRA:
python benchmarks/dpo/dpo_continual.py \
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
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from mock_data import init_mock_dataset


def main(
    script_args: ScriptArguments, training_args: DPOConfig, model_args: ModelConfig
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

    continual_dataset = init_mock_dataset(script_args.dataset_name)
    print('training_args', training_args)
    print('model_args', model_args)
    print('script_args', script_args)
    output_dir = training_args.output_dir
    for i, dataset in enumerate(continual_dataset.datasets):
        training_args.output_dir = f'{output_dir}/dataset-{i}'
        trainer = DPOTrainer(
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

        trainer.train()

        if training_args.eval_strategy != 'no':
            metrics = trainer.evaluate()
            metrics['dataset'] = i + 1
            trainer.log_metrics('eval' + f'_dataset-{i}', metrics)
            trainer.save_metrics('eval' + f'_dataset-{i}', metrics)
            metrics = {'last/' + k: v for k, v in metrics.items()}
            wandb.log(metrics)

        # Save and push to hub
        trainer.save_model(training_args.output_dir + '/last')
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=('Continual_DPO_' + script_args.dataset_name + f'/dataset-{i}'))


if __name__ == '__main__':
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)

    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
