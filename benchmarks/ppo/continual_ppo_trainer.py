import functools
import inspect
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    default_data_collator,
)
from transformers.trainer_callback import TrainerCallback
from trl import ScriptArguments
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.ppo_trainer import (
    PPOTrainer,
    batch_generation,
    get_reward,
    unwrap_model_for_generation,
)


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
        eval_policy_dataloader: Optional[DataLoader] = None,
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
        self.eval_policy_dataloader = eval_policy_dataloader

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

    def get_eval_dataloader(self) -> DataLoader:
        """Returns a DataLoader for the evaluation dataset.
        If eval_dataset is a dict, the first dataset in the dict is chosen.
        Uses self.args.eval_batch_size if it exists, defaulting to 8 otherwise.
        Uses self.data_collator if provided, otherwise uses the default data collator.
        """
        if self.eval_dataset is None:
            raise ValueError('No evaluation dataset provided.')
        batch_size = getattr(self.args, 'eval_batch_size', 8)
        if isinstance(self.eval_dataset, dict):
            # Use the first dataset available in the dict
            eval_dataset = list(self.eval_dataset.values())[0]
        else:
            eval_dataset = self.eval_dataset

        # Use the provided data_collator if available; otherwise, fall back to default_data_collator.
        collate_fn = (
            self.data_collator
            if self.data_collator is not None
            else default_data_collator
        )

        return DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

    def evaluate(self) -> dict:
        """Custom evaluation method for PPO. Generates completions from the evaluation dataloader,
        computes rewards using the reward model, and returns aggregated metrics.
        """
        mode: bool = self.model.training
        self.model.eval()
        eval_metrics = defaultdict(list)
        processing_class = self.processing_class

        # Configure generation settings (using either greedy or sampling strategy)
        if self.args.eval_greedy_policy:
            generation_config = GenerationConfig(
                max_new_tokens=self.args.response_length,
                top_k=None,
                do_sample=False,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=self.args.response_length,
                temperature=self.args.temperature + 1e-7,
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
            )

        # Ensure eval_policy_dataloader exists
        if (
            not hasattr(self, 'eval_policy_dataloader')
            or self.eval_policy_dataloader is None
        ):
            self.eval_policy_dataloader = self.get_eval_dataloader()

        with torch.no_grad():
            for batch in self.eval_policy_dataloader:
                # Move the query tokens to the correct device
                query = batch['input_ids'].to(self.accelerator.device)
                context_length = query.shape[1]

                with unwrap_model_for_generation(
                    self.model, self.accelerator
                ) as unwrapped_model:
                    core_model = unwrapped_model
                    # Recursively unwrap until we get a model with a generate() method.
                    while not hasattr(core_model, 'generate'):
                        if hasattr(core_model, 'policy'):
                            core_model = core_model.policy
                        elif hasattr(core_model, 'model'):
                            core_model = core_model.model
                        elif hasattr(core_model, 'policy_model'):
                            core_model = core_model.policy_model
                        else:
                            break  # Cannot unwrap further; exit loop.

                    query_response, _ = batch_generation(
                        core_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )

                    response = query_response[:, context_length:]

                # Combine the original query with the generated response and calculate the reward
                postprocessed_query_response = torch.cat((query, response), dim=1)
                _, score, _ = get_reward(
                    self.reward_model,
                    postprocessed_query_response,
                    processing_class.pad_token_id,
                    context_length,
                )
                # Gather scores from all processes and append to our metrics
                eval_metrics['score'].extend(
                    self.accelerator.gather_for_metrics(score).float().cpu().numpy()
                )

        self.model.train(mode)
        # Aggregate and return the metrics (here, averaging the reward scores)
        return {'eval_' + k: float(np.mean(v)) for k, v in eval_metrics.items()}

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ) -> None:
        """Save the model, dealing with the case where it's a PEFT model without a policy attribute."""
        # Store the original model
        original_model = self.model

        # For PEFT models (which lack .policy attribute), use the model directly
        if hasattr(self.model, 'base_model'):
            # PEFT model case - don't try to access .policy
            pass  # Keep the model as is
        elif hasattr(self.model, 'policy'):
            # Standard PPO case - use the policy as in the original implementation
            self.model: nn.Module = self.model.policy
        elif hasattr(self.model, 'policy_model'):
            # Standard PPO case - use the policy_model as in the original implementation
            self.model = self.model.policy_model

        # Call the parent class's save_model
        if output_dir is None:
            output_dir = self.args.output_dir

        Trainer.save_model(self, output_dir, _internal_call)

        # Restore the original model
        self.model = original_model
