#!/bin/bash
#SBATCH --job-name=validate_static_all_diversity
#SBATCH --partition=unkillable-cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

source .env

# list all sub‐tasks
tasks=(
  education_qna_direct
  education_qna_eli5
  education_qna_expert
  education_qna_hinted
  education_summary_eli5
  education_summary_expert
  politics_generate_formal
  politics_generate_rapper
  politics_generate_shakespeare
  politics_qna_eli5
  politics_qna_expert
  politics_summary_eli5
  politics_summary_expert
  tech_healthcare_qna_eli5
  tech_healthcare_qna_expert
  tech_physics_summary_eli5
  tech_physics_summary_expert
)

for t in "${tasks[@]}"; do
  echo "Validating $t..."
  uv run aif validate \
    "data/4omini_generation/$t/data.json" \
    "data/4omini_validation_no_diversity/$t/validate.json" \
    --no-validate-diversity \
    --no-validate-count \
    --no-validate-entropy \
    --no-validate-llm-judge \
    --embedding-model "bge-m3" \
    --embedding-batch-size 256 \
    --max-concurrency 16 \
    || { echo "Validation failed on $t"; exit 1; }
done

echo "All validations completed successfully."
