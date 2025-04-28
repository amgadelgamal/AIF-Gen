#!/bin/bash
#SBATCH --job-name=validate_static_all_diversity
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shahrad_m@icloud.com

cd
module load python/3.10
module load cuda/12.6.0
source .venv/bin/activate
echo "Starting vLLM server..."
uv run vllm serve BAAI/bge-m3 --dtype float16 --api-key openai --task embed &

# Save server process ID
SERVER_PID=$!

echo "Waiting for server to start..."
while true; do
  echo "Checking if server is up..."
  RESPONSE=$(curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" 2>&1)

  if [[ "$RESPONSE" == *"data"* ]]; then
    echo "Server is up and running!"
    break
  fi

  # Check if server is still running
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server process died unexpectedly"
    exit 1
  fi

  echo "Server not ready yet. Waiting 5 seconds..."
  sleep 5
done

deactivate
cd projects/AIF-Gen
source .env

echo "Starting validation process..."

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
  tech_physics_summary_highschool
)

for t in "${tasks[@]}"; do
  echo "Validating $t..."
  uv run aif validate \
    "data/4omini_generation/$t/data.json" \
    "data/4omini_validation_diversity/$t/validate.json" \
    --no-validate-diversity \
    --no-validate-count \
    --no-validate-entropy \
    --no-validate-llm-judge \
    --embedding-model "BAAI/bge-m3" \
    --embedding-batch-size 256 \
    --max_concurrency 16 \
    || { echo "Validation failed on $t"; exit 1; }
done

echo "All validations completed successfully."
