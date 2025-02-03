import tyro
import torch
import random
import numpy as np
import wandb

from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from dataclasses import dataclass

from aif_gen.dataset import DebugContinualDataset, ContinualUltrafeedback2AnthropicDataset


@dataclass
class Args:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct" # facebook/opt-350m gpt2 google/flan-t5-small
    """base model to start to finetune with"""
    output_dir: str = "~/scratch/DPO"
    """output directory"""
    dataset: str = "debug"
    """dataset to use, current options are: debug and ultrafeedback2anthropic"""
    exp_name: str = "default"
    """experiment name"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    seed: int = 1
    """seed of the experiment"""
    wandb_project_name: str = "AIF-Gen-Training"
    """the wandb's project name"""
    wandb_entity: str = ''
    """the entity (team) of wandb's project"""
    num_train_epochs: int = 1
    """number of training epochs"""
    per_device_train_batch_size: int = 4
    """batch size per device"""
    wandb_mode: str = "online"
    """wandb mode, either online or offline"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    group_name = f"{args.model_name}-DPO-{args.exp_name}"
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
    model.warnings_issued = {}
    model.config.return_dict = True

    # Apply LoRA to the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set the tokenizer's pad_token to eos_token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == 'debug':
        continual_dataset = DebugContinualDataset()
    elif args.dataset == 'ultrafeedback2anthropic':
        continual_dataset = ContinualUltrafeedback2AnthropicDataset()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    training_args = DPOConfig(output_dir=args.output_dir,
                              per_device_train_batch_size=args.per_device_train_batch_size,
                              num_train_epochs=args.num_train_epochs,
                              evaluation_strategy="epoch",
                              per_device_eval_batch_size=args.per_device_train_batch_size,
                              lr_scheduler_type="constant",
                              )

    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, group=group_name, name=run_name,
               config=vars(training_args), mode=args.wandb_mode)

    for i, dataset in enumerate(continual_dataset.datasets):
        trainer = DPOTrainer(model=model, args=training_args,
                             train_dataset=dataset['train'],
                             eval_dataset=dataset['test'],
                             processing_class=tokenizer,
                             )

        print('running evaluation on dataset', i)
        eval_results = trainer.evaluate()
        eval_results = {"f/"+k: v for k, v in eval_results.items()}
        # ToDo: maybe it's better to log the dataset name or index, that should come from the dataset itself
        eval_results['dataset'] = i
        wandb.log(eval_results)

        print('running training on dataset', i)
        trainer.train()

        # Save the model
        save_path = f"{args.output_dir}/dataset-{i}/{run_name}"
        trainer.save_model(save_path)

    print('running evaluation on dataset', i)
    eval_results = trainer.evaluate()
    eval_results = {"f/" + k: v for k, v in eval_results.items()}
    eval_results['dataset'] = i + 1
    wandb.log(eval_results)

    wandb.finish()