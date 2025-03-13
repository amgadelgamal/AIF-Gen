# Dataset Generation

(Temporary) folder for dataset generation utilities.

Example usage:

```bash
# .env
export OPENAI_BASE_URL="..."
export OPENAI_API_KEY="..."
```

```bash
source .env && uv run aif_gen/generate/entrypoint.py
```

Output will be saved under the folder `data/`.
