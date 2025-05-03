#!/bin/bash
#SBATCH --job-name=validate_static_diversity
#SBATCH --tasks=1
#SBATCH --cpus-per-task=6
#SBATCH --account=
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

cd
module load python/3.10
module load cuda/12.6
source .venv/bin/activate
echo "Starting vLLM server..."
uv run vllm serve Salesforce/SFR-Embedding-Mistral \
            --dtype bfloat16 \
            --api-key openai \
            --kv-cache-dtype fp8 \
            --task embed \
            --trust-remote-code \
            --tensor_parallel_size 2 \
            --max-model-len 4096 &

# Save server process ID
SERVER_PID=$!

echo "Waiting for server to start..."
while true; do
  echo "Checking if server is up..."
  RESPONSE=$(curl -s http://localhost:8000/v1/models -H "Authorization: Bearer openai" 2>&1)

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
  merged_qna
  merged_qna_summary
  hh
  ultra
  cppo-rl-sampled
  cppo-reward-sampled
)

for t in "${tasks[@]}"; do
  echo "Validating $t..."
  uv run aif validate \
    "data/$t.json" \
    "data/$t-validate-diversity-nvidia.json" \
    --no-validate-diversity \
    --no-validate-count \
    --no-validate-entropy \
    --no-validate-llm-judge \
    --embedding-model "Salesforce/SFR-Embedding-Mistral" \
    --embedding-batch-size 128 \
    --max_concurrency 8 \
    || { echo "Validation failed on $t"; exit 1; }
done

echo "All validations completed successfully."
