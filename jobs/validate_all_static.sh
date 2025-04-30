#!/bin/bash
#SBATCH --job-name=validate_static_all
#SBATCH --partition=unkillable-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

source .env

MODEL="gpt-4o-mini"
MAX_CONC=256

# list all sub‐tasks
tasks=(
  ultra-hh-sampled
)

for t in "${tasks[@]}"; do
  echo "Validating $t..."
  uv run aif validate \
    --max_concurrency "$MAX_CONC" \
    "data/$t.json" \
    "data/$t-validate-no-diversity.json" \
    --no-validate-diversity \
    --no-validate-embedding-diversity \
    --model "$MODEL" \
    || { echo "Validation failed on $t"; exit 1; }
done

echo "All validations completed successfully."
