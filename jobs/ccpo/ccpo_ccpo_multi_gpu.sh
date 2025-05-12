#!/bin/bash
#SBATCH --job-name=aif-gen-cppo-cppo
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=out/%j.out
#SBATCH --error=out/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

source .env

dataset_name='CPPO-RL'

accelerate launch --config_file benchmarks/cppo/accelerate_configs/deepspeed_zero2.yaml \
    benchmarks/cppo/cppo.py \
    --wandb_project $dataset_name \
    --wandb_run_name "Qwen2-0.5B-CPPO-${dataset_name}-multi-gpu" \
    --dataset_name $dataset_name \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path Shahradmz/Qwen2-0.5B-Instruct_${dataset_name}_REWARD_0 \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_${dataset_name}_REWARD \
    --learning_rate 5.0e-6 \
    --response_length 256 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 4 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_steps 10 \
    --bf16 \
    --output_dir "$SCRATCH/Qwen2-0.5B-CPPO-${dataset_name}" \
    --no_remove_unused_columns
