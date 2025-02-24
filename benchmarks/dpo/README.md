# Adaptation of TRL for Continual Learning

### Sync additional dependencies

```sh
uv sync --group benchmarks.dpo
```

### Run DPO

```sh
uv run baselines/trl/dpo_continual.py \
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
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
    --dataset_name debug
    --wandb_project qwen_test
```

- In order to run with accelerate or without lora, refer to `dpo_continual.py` for more details.

### Run a sweep with wandb

```sh
wandb sweep sweep_configs/dpo_sweep.yaml    # which returns the SWEEP_ID
```

and

```sh
wandb agent <SWEEP_ID>
```

- All details per task and hyperparameters are going to be loaded in your wandb dashboard.
