# Adaptation of TRL's reward modeling integrated with MAS for continual learning

### Sync additional dependencies

```sh
uv sync --group benchmarks
```

## Reward Modeling

### Full Training

```sh
uv run benchmarks/reward_mas/reward_modeling_mas.py \
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
    --max_length 2048 \
    --mas_lambda 1.0 \
    --push_to_hub True
```

### With LoRA and Push to Hub

```sh
uv run benchmarks/reward_mas/reward_modeling_mas.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_index 0 \
    --dataset_name benchmarks/continual_data_debug.json \
    --output_dir "$SCRATCH/Qwen2-0.5B-Reward-LoRA" \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 30 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --push_to_hub True \
    --wandb_project Qwen2-0.5B-Reward-LoRA_test \
    --wandb_run_name dataset_0 \
    --mas_lambda 1.0
```

All these commands ensure that whether you run full training or use LoRA and pushing to the Hub, the parameters remain consistent with your working configuration. Adjust any parameters as needed in your own setup.

### Citation

```bibtex
@inproceedings{aljundi2018memory,
  title={Memory aware synapses: Learning what (not) to forget},
  author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={139--154},
  year={2018}
}
```
