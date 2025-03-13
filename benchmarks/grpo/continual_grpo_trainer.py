import functools
import inspect
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from trl import GRPOConfig, ScriptArguments
from trl.trainer.ppo_trainer import PPOTrainer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO training script.

    Args:
        reward_model_path (`str` or `None`):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
    """

    reward_model_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Reward model id of a pretrained model hosted inside a model repo on huggingface.co or '
            'local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`.'
        },
    )
    dataset_name: str = field(
        default='debug',
        metadata={'help': 'The name or path of the continual dataset to use.'},
    )
    wandb_project: Optional[str] = field(
        default='AIFGen-ppo-continual-test',
        metadata={'help': 'Override the default WandB project name.'},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={'help': 'The WandB entity (team) to use.'},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={'help': 'The WandB run name.'},
    )

    def __post_init__(self) -> None:
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity


@dataclass
class ContinualGRPOConfig(GRPOConfig):
    mock: bool = field(
        default=False,
        metadata={'help': 'Whether to use mock dataset.'},
    )
    eval_greedy_policy: bool = field(
        default=False,
        metadata={'help': 'Whether to use greedy policy for evaluation.'},
    )


class ContinualGRPOTrainer(PPOTrainer):
    # Shared accelerator instance across all trainer instances
    shared_accelerator: Optional[Accelerator] = None
    accelerator: Accelerator  # now non-optional after creation

    def __init__(
        self,
        args: Optional[ContinualGRPOConfig] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        peft_config: Optional[dict] = None,
    ):
        # catching this here to test our implementation of the configs
        if args is None:
            raise ValueError('`args` cannot be None')

        if ContinualGRPOTrainer.shared_accelerator is None:
            ContinualGRPOTrainer.shared_accelerator = Accelerator(
                gradient_accumulation_steps=args.gradient_accumulation_steps
            )
        self.accelerator = ContinualGRPOTrainer.shared_accelerator

        super().__init__(
            args=args,
            processing_class=processing_class,
            model=model,
            reward_funcs=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )

        # No need for anything else as PPO itself is already set up with the reward model
        self.accelerator = (
            ContinualGRPOTrainer.shared_accelerator
        )  # turn the accelerator back to the shared one

    def create_accelerator_and_postprocess(self) -> None:
        # Only initialize a new Accelerator if one does not exist
        if ContinualGRPOTrainer.shared_accelerator is None:
            super().create_accelerator_and_postprocess()
            ContinualGRPOTrainer.shared_accelerator = self.accelerator
        else:
            # Reuse the shared accelerator
            self.accelerator = ContinualGRPOTrainer.shared_accelerator
            self.gather_function = self.accelerator.gather_for_metrics
            if (
                'use_gather_object'
                in inspect.signature(self.gather_function).parameters.keys()
            ):
                self.gather_function = functools.partial(
                    self.gather_function,
                    use_gather_object=self.args.eval_use_gather_object,
                )
            self.is_deepspeed_enabled = (
                getattr(self.accelerator.state, 'deepspeed_plugin', None) is not None
            )
            self.is_fsdp_enabled = (
                getattr(self.accelerator.state, 'fsdp_plugin', None) is not None
            )
