#!/bin/bash
#SBATCH --job-name=generate_static
#SBATCH --partition=unkillable-cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shahrad_m@icloud.com

source .env
uv run aif generate config/static/education_qna_hinted.yaml \
                    $GENERATION_MODEL_NAME \
                    --output_file "$SCRATCH"/education_qna_hinted.json \
                    --hf-repo-id Shahradmz/education_qna_hinted \
                    --random_seed $RANDOM_SEED
