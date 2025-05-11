#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:1
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --job-name=aifgen
#SBATCH --partition=main
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

source .env

dataset_name=${1:-'aifgen-lipschitz'}

python benchmarks/ppo/ppo_continual.py \
    --dataset_name $dataset_name \
    --sft_model_path Qwen/Qwen2-0.5B-Instruct \
    --value_model_path "LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD_0" \
    --reward_model_path "LifelongAlignment/Qwen2-0.5B-Instruct_${dataset_name}_REWARD" \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --bf16 \
    --output_dir "$SCRATCH"/Qwen2-0.5B-PPO \
    --no_remove_unused_columns \
    --push_to_hub False
