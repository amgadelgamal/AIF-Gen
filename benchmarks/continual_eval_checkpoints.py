"""Evaluating checkpoints obtained from training using the dpo_continual script.

ONLY EVALUATION

wandb init --entity your_entity --project aifgen

LoRA:
python benchmarks/dpo/continual_eval_checkpoints.py \
    --dataset_name  debug \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --checkpoint_dir Qwen2-0.5B-DPO-test \
    --learning_rate 0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 1000 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --bf16 \
    --output_dir Qwen2-0.5B-DPO-test-eval \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import glob
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from continual_dpo_trainer import ContinualDPOTrainer
from dataloading import init_continual_dataset
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


@dataclass
class EvalScriptArguments(ScriptArguments):
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory containing the checkpoints to evaluate.'},
    )
    dataset_name: str = field(
        default='debug',
        metadata={'help': 'The name or path of the continual dataset to use.'},
    )


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

    continual_dataset: list[dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name
    )
    output_dir = training_args.output_dir

    checkpoint_paths = glob.glob(f'{script_args.checkpoint_dir}/*/*')
    checkpoint_paths = sorted([ch for ch in checkpoint_paths if 'checkpoint' in ch])

    for checkpoint_path in checkpoint_paths:
        dataset_name = checkpoint_path.split('/')[-2]
        checkpoint_step = checkpoint_path.split('/')[-1]
        adapter_name = dataset_name + checkpoint_step
        model.load_adapter(checkpoint_path, adapter_name=adapter_name)
        metrics = {}
        for i, dataset in enumerate(continual_dataset):
            training_args.output_dir = f'{output_dir}/dataset-{i}'
            trainer = ContinualDPOTrainer(
                model,
                ref_model,
                args=training_args,
                train_dataset=dataset[script_args.dataset_test_split],
                eval_dataset=dataset[script_args.dataset_test_split],
                processing_class=tokenizer,
                peft_config=peft_config,
            )

            ev_metrics = trainer.evaluate()
            ev_metrics = {f'dataset-{i}/' + k: v for k, v in ev_metrics.items()}
            if i == 0:
                # log the name of the dataset
                trainer.log({'dataset_name': dataset_name})
            metrics.update(ev_metrics)

        wandb.log(metrics)


if __name__ == '__main__':
    dataclass_types = (EvalScriptArguments, DPOConfig, ModelConfig)
    parser = TrlParser(dataclass_types)

    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
