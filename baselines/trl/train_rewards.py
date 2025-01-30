from datasets import load_dataset
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer
from peft import LoraConfig
from dataclasses import dataclass
import tyro
import torch
import random
import numpy as np

import wandb


@dataclass
class Args:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct" # facebook/opt-350m gpt2 google/flan-t5-small
    """model name"""
    dataset_idx: int = 0
    """dataset index"""
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # ToDo: replace with our actual datasets, define the eval datasets as well
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test").select(range(20))

    group_name = f"{args.model_name}-Rewards-{args.dataset_idx}"
    run_name = f"{group_name}-{args.seed}"

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = RewardConfig(output_dir=group_name,
                                 per_device_train_batch_size=args.per_device_train_batch_size,
                                 num_train_epochs=args.num_train_epochs)

    # ToDo: log the dataset name or index, that should come from the dataset itself
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
               group=group_name, name=run_name, config=vars(training_args))
    trainer = RewardTrainer(
        args=training_args,
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
    )
    trainer.train()
    eval_results = trainer.evaluate()
    wandb.log(eval_results)
    trainer.save_model(f"Reward_Models/{group_name}/{run_name}-dataset-{args.dataset_idx}")

    wandb.finish()