# Adaptation of TRL for Continual Learning

### Sync additional dependencies

```sh
uv sync --group benchmarks.dpo
```

## Overview

`dpo/dpo_continual` is the primary script for training DPO (Direct Preference Optimization) in a continual learning setting.

The wandb run gets `huggingface` project name and run name from the `output_dir` argument. The `output_dir` argument is used to save the model and logs.

## Components

- **`continual_dpo_trainer`**: Defines modifications of TRL `DPOTrainer` and arguments specific to continual learning.
- **`reward_modeling`**: Handles training of the reward model. Each reward model is trained independently with each continual learning subtask.
- **`continual_eval_checkpoints`**: Evaluates model checkpoints after training with `dpo_continual`, ensuring that training time is focused on learning rather than evaluation.

## Reward Modeling

### Full training

```sh
uv run benchmarks/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name benchmarks/continual_data_debug.json \
    --dataset_index 0 \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048
```

### Lora

```sh
uv run benchmarks/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name benchmarks/continual_data_debug.json \
    --output_dir Qwen2-0.5B-Reward-LoRA \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```

## Evaluation

### Lora

```sh
uv run benchmarks/continual_eval_checkpoint.py \
    --dataset_name benchmarks/continual_dpo_trainer \
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
```
