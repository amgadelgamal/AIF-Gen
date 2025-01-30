from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RLOOConfig, RLOOTrainer, apply_chat_template
from trl import DPOConfig, DPOTrainer, AutoModelForCausalLMWithValueHead
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
    """base model to start to finetune with"""
    output_dir: str = "~/scratch/PPO"
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

    group_name = f"{args.model_name}-PPO"
    run_name = f"{group_name}-{args.seed}"


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ref_policy = AutoModelForCausalLM.from_pretrained(args.model_name)
    policy = AutoModelForCausalLM.from_pretrained(args.model_name)

    # ToDo: replace with our actual datasets, define the eval datasets as well
    dataset = load_dataset("trl-lib/ultrafeedback-prompt")['train'].select(range(20))
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.map(lambda x: tokenizer(x["prompt"]), remove_columns="prompt")
    datasets = [
        dataset,
        dataset,
                ]

    training_args = RLOOConfig(output_dir=args.output_dir,
                               per_device_train_batch_size=args.per_device_train_batch_size,
                               num_train_epochs=args.num_train_epochs,
                               evaluation_strategy="epoch",
                               per_device_eval_batch_size=args.per_device_train_batch_size,
                               )

    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, group=group_name, name=run_name, config=vars(training_args))

    for i, dataset in enumerate(datasets):

        # ToDo: replace with our actual reward model that correspods to the correct dataset i
        reward_model_path = "Reward_Models/Qwen/Qwen2.5-0.5B-Instruct-Rewards-0/Qwen/Qwen2.5-0.5B-Instruct-Rewards-0-1-dataset-0"
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, num_labels=1)

        trainer = RLOOTrainer(
            config=training_args,
            processing_class=tokenizer,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            train_dataset=dataset,
            eval_dataset=dataset,
        )

        # eval_results = trainer.evaluate()
        # eval_results = {"f_"+k: v for k, v in eval_results.items()}
        # # ToDo: log the dataset name or index, that should come from the dataset itself
        # eval_results['dataset'] = i
        # wandb.log(eval_results)

        trainer.train()

        # Save the model
        save_path = f"{args.output_dir}/dataset-{i}/{run_name}"
        trainer.save_model(save_path)

    # eval_results = trainer.evaluate()
    # eval_results = {"f_" + k: v for k, v in eval_results.items()}
    # eval_results['dataset'] = i
    # wandb.log(eval_results)

    wandb.finish()