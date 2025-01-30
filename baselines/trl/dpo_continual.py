from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from dataclasses import dataclass
import tyro
import torch
import random
import numpy as np

import wandb

# ToDo: dummy data, delete later
from datasets import Dataset
data = {
    "prompt": [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Tell me a joke.",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Tell me a joke."
    ],
    "chosen": [
        "The capital of France is Paris.",
        "Quantum computing uses quantum bits (qubits) that can exist in multiple states at once, allowing for complex calculations.",
        "Why don’t skeletons fight each other? Because they don’t have the guts!",
        "The capital of France is Paris.",
        "Quantum computing uses quantum bits (qubits) that can exist in multiple states at once, allowing for complex calculations.",
        "Why don’t skeletons fight each other? Because they don’t have the guts!"
    ],
    "rejected": [
        "France is a country in Europe.",
        "It's a way of using computers differently.",
        "I don't know any jokes.",
        "France is a country in Europe.",
        "It's a way of using computers differently.",
        "I don't know any jokes."
    ],
}


@dataclass
class Args:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct" # facebook/opt-350m gpt2 google/flan-t5-small
    """base model to start to finetune with"""
    output_dir: str = "~/scratch/DPO"
    """output directory"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    seed: int = 1
    """seed of the experiment"""
    wandb_project_name: str = "conitnual_finetuning_baselines"
    """the wandb's project name"""
    wandb_entity: str = 'irina-rish'
    """the entity (team) of wandb's project"""
    num_train_epochs: int = 1
    """number of training epochs"""
    per_device_train_batch_size: int = 4
    """batch size per device"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    group_name = f"{args.model_name}-DPO"
    run_name = f"{group_name}-{args.seed}"

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name, peft_config=lora_config,)
    # print('model:', list(model.pretrained_model.model.base_model.layers[-1].self_attn.q_proj.lora_A.parameters()))

    model.warnings_issued = {}
    model.config.return_dict = True

    # Apply LoRA to the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set the tokenizer's pad_token to eos_token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ToDo: replace with our actual datasets, define the eval datasets as well
    train_datasets = [
        # Dataset.from_dict(data),
        # Dataset.from_dict(data),
        # load_dataset("trl-lib/ultrafeedback_binarized", split="test").select(range(10)),
        # load_dataset("trl-lib/ultrafeedback_binarized", split="test").select(range(10, 20)),
        # load_dataset("Anthropic/hh-rlhf", split="test").select(range(10))
        load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
        load_dataset("Anthropic/hh-rlhf", split="train")
                ]

    test_datasets = [
        load_dataset("trl-lib/ultrafeedback_binarized", split="test"),
        load_dataset("Anthropic/hh-rlhf", split="test")
                ]

    training_args = DPOConfig(output_dir=args.output_dir,
                              per_device_train_batch_size=args.per_device_train_batch_size,
                              num_train_epochs=args.num_train_epochs,
                              evaluation_strategy="epoch",
                              per_device_eval_batch_size=args.per_device_train_batch_size,
                              )

    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, group=group_name, name=run_name, config=vars(training_args))

    for i, (train_dataset, test_dataset) in enumerate(zip(train_datasets, test_datasets)):
        trainer = DPOTrainer(model=model, args=training_args,
                             train_dataset=train_dataset,
                             eval_dataset=test_dataset,
                             processing_class=tokenizer,
                             )

        print('running evaluation on dataset', i)
        eval_results = trainer.evaluate()
        eval_results = {"f_"+k: v for k, v in eval_results.items()}
        # ToDo: log the dataset name or index, that should come from the dataset itself
        eval_results['dataset'] = i
        wandb.log(eval_results)

        print('running training on dataset', i)
        trainer.train()

        # Save the model
        save_path = f"{args.output_dir}/dataset-{i}/{run_name}"
        trainer.save_model(save_path)

    print('running evaluation on dataset', i)
    eval_results = trainer.evaluate()
    eval_results = {"f_" + k: v for k, v in eval_results.items()}
    eval_results['dataset'] = i
    wandb.log(eval_results)

    wandb.finish()