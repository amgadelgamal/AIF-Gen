#!/bin/bash
#SBATCH --job-name=generate_static_formal
#SBATCH --partition=unkillable-cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

source .env
uv run aif generate --max_concurrency 256 \
                    config/static/politics_generate_formal.yaml \
                    gpt-4o-mini
