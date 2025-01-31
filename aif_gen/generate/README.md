# Dataset Generation

(Temporary) folder for dataset generation utilities.

Example usage:

```bash
# .env
export OPENAI_BASE_URL="..."
export OPENAI_API_KEY="..."
```

```bash
source .env && python3 -m aif_gen.generate.entrypoint --model_name Meta-Llama-3.1-8B-Instruct
```

Output will be saved under the folder `data/`.


## Known Limitations

Structured decoding in vLLM appears to be significantly slower than regular, unconstrained decoding. 

See [vllm #11908](https://github.com/vllm-project/vllm/issues/11908).