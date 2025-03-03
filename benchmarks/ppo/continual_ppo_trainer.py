import functools
import inspect
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from trl import ScriptArguments
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.ppo_trainer import PPOTrainer


@dataclass
class ContinualPPOArguments(ScriptArguments):
    value_model_path: str = field(
        default='AIF-Gen/Qwen/Qwen2-0.5B-Reward/debug/0',
        metadata={'help': 'Path to the value model or a HuggingFace model path.'},
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

    def __post_init__(self) -> None:
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity


@dataclass
class ContinualPPOConfig(PPOConfig):
    mock: bool = field(
        default=False,
        metadata={'help': 'Whether to use mock dataset.'},
    )
    response_length: int = field(
        default=53,
        metadata={
            'help': 'Length of the response. Borrowed from PPOCConfig and used only for evaluation.'
        },
    )
    temperature: float = field(
        default=0.7,
        metadata={
            'help': 'Temperature for sampling. Borrowed from PPOConfig and used only for evaluation, taken from OnPolicyConfig config'
        },
    )
    eval_greedy_policy: bool = field(
        default=False,
        metadata={'help': 'Whether to use greedy policy for evaluation.'},
    )


class ContinualPPOTrainer(PPOTrainer):
    # Shared accelerator instance across all trainer instances
    shared_accelerator: Optional[Accelerator] = None
    accelerator: Accelerator  # now non-optional after creation

    def __init__(
        self,
        args: Optional[PPOConfig] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        train_dataset: Optional[Dataset] = None,
        value_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional[dict] = None,
    ):
        # catching this here to test our implementation of the configs
        if args is None:
            raise ValueError('`args` cannot be None')
        super().__init__(
            args,
            processing_class,
            model,
            ref_model,
            reward_model,
            train_dataset,
            value_model,
            data_collator,
            eval_dataset,
            optimizers,
            callbacks,
            peft_config,
        )

        # No need for anything else as PPO itself is already set up with the reward model

    def create_accelerator_and_postprocess(self) -> None:
        # Only initialize a new Accelerator if one does not exist
        if ContinualPPOTrainer.shared_accelerator is None:
            super().create_accelerator_and_postprocess()
            ContinualPPOTrainer.shared_accelerator = self.accelerator
        else:
            # Reuse the shared accelerator
            self.accelerator = ContinualPPOTrainer.shared_accelerator
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
