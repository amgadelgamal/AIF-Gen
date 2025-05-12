#!/bin/bash
#SBATCH --job-name=cppo_debug_multi_gpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100l:2
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

source .env

accelerate launch --config_file benchmarks/cppo/accelerate_configs/deepspeed_zero2.yaml \
    benchmarks/cppo/cppo.py \
    --dataset_name benchmarks/continual_data_debug.json \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD_0 \
    --reward_model_path Shahradmz/Qwen2-0.5B-Instruct_continual_data_debug_REWARD \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --per_device_train_batch_size 2 \
    --logging_steps 2 \
    --eval_strategy steps \
    --eval_steps 5 \
    --save_steps 5 \
    --bf16 \
    --output_dir "$SCRATCH/Qwen2-0.5B-CPPO-test" \
    --no_remove_unused_columns
