# Adaptation of TRL for Continual Learning

### Sync additional dependencies

```sh
uv sync --group benchmarks.ppo
```

## Run PPO

### Lora

```sh
uv run benchmarks/ppo/ppo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --mock False \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path Qwen/Qwen2-0.5B-Reward/debug \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 20 \
    --bf16 \
    --output_dir Qwen2-0.5B-PPO-test \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```

```sh
uv run baselines/trl/ppo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --mock False \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --total_episodes 20 \
    --use_peft \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path /home/mila/i/ivan.anokhin/AIF-Gen/Qwen/Qwen2-0.5B-Reward/debug \
    --missing_eos_penalty 1.0
```

### Lora with accelerate

```sh
accelerate launch --config_file benchmarks/ppo/accelerate_configs/deepspeed_zero3.yaml \
    benchmarks/ppo/ppo_continual.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --mock False \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path Qwen/Qwen2-0.5B-Reward/debug \
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
    --output_dir Qwen2-0.5B-PPO-test \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```

### Full training

```sh
uv run benchmarks/ppo/ppo_continual.py \
    --dataset_name debug \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --reward_model_path Qwen/Qwen2-0.5B-Reward/debug \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-PPO \
    --no_remove_unused_columns
```

### Run a sweep with wandb

```sh
wandb sweep sweep_configs/ppo_sweep.yaml    # which returns the SWEEP_ID
```

and

```sh
wandb agent <SWEEP_ID>
```

- All details per task and hyperparameters are going to be loaded in your wandb dashboard.
