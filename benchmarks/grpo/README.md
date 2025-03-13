# Adaptation of TRL for Continual Learning

This repository adapts TRL for continual learning. The commands below use a consistent set of parameters that you’ve identified as working. You can use any of the entrypoints (uv run, accelerate launch, wandb) with the following commands.

### Sync additional dependencies

```sh
uv sync --group benchmarks.ppo
```

## Run GRPO

### Using uv run (vanilla Python with PEFT)

```sh
uv run benchmarks/grpo/grpo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 20 \
    --bf16 \
    --output_dir "$SCRATCH/Qwen2-0.5B-PPO-test" \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```

```sh
python benchmarks/grpo/grpo_continual.py \
    --dataset_name debug \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/grpo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --use_peft \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path /home/mila/i/ivan.anokhin/AIF-Gen/Qwen/Qwen2-0.5B-Reward/debug
```
