from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, AutoModelForCausalLMWithValueHead

from peft import LoraConfig

import wandb

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
output_dir = f"{model_name}-DPO"

# LoRA configuration
lora_config = LoraConfig(
    r=8,               # Low-rank dimension
    lora_alpha=16,     # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA
    lora_dropout=0.05, # Dropout rate
    bias="none"        # Whether to adapt biases
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, peft_config=lora_config,)

# model = AutoModelForCausalLM.from_pretrained(model_name)

model.warnings_issued = {}
model.config.return_dict = True

# Apply LoRA to the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test")
# dataset = load_dataset("Anthropic/hh-rlhf", split="test")
# dataset = load_dataset('openai/summarize_from_feedback', 'axis', split='validation')

training_args = DPOConfig(output_dir=output_dir)
training_args.per_device_train_batch_size = 4
training_args.num_train_epochs = 1

trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset, eval_dataset=dataset, processing_class=tokenizer)

wandb.init(project='conitnual_finetuning_baselines', name=output_dir)
trainer.train()