#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --job-name=aifgen-dpo-ewc-piecewise-preference-shift
#SBATCH --partition=main
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

source .env

dataset_name='aifgen-piecewise-preference-shift'

python benchmarks/dpo_ewc/dpo_EWC_continual.py \
    --wandb_project $dataset_name \
    --wandb_run_name "Qwen2-0.5B-DPO-EWC_${dataset_name}" \
    --dataset_name $dataset_name \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
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
    --output_dir "$SCRATCH"/Qwen2-0.5B-DPO-EWC-${dataset_name} \
    --no_remove_unused_columns
