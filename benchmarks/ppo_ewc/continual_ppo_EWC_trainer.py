from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
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
from trl.trainer.ppo_config import PPOConfig

from benchmarks.ppo.continual_ppo_trainer import (
    ContinualPPOArguments,
    ContinualPPOConfig,
    ContinualPPOTrainer,
)


@dataclass
class ContinualPPOEWCArguments(ContinualPPOArguments):
    """Arguments for Continual PPO training with EWC regularization."""

    wandb_project: Optional[str] = field(
        default='AIFGen-ppo-EWC-continual-test',
        metadata={'help': 'Override the default WandB project name for EWC runs.'},
    )


@dataclass
class ContinualPPOEWCConfig(ContinualPPOConfig):
    """Configuration for Continual PPO training with EWC regularization."""

    ewc_lambda: float = field(
        default=100.0,
        metadata={
            'help': 'EWC regularization strength. Higher values give stronger regularization.'
        },
    )
    ewc_importance_decay: float = field(
        default=0.5,
        metadata={
            'help': 'Decay factor for previous task importance (0-1). 0 means only care about most recent task.'
        },
    )
    fisher_estimation_samples: int = field(
        default=200,
        metadata={
            'help': 'Number of samples to use when estimating Fisher information matrix'
        },
    )


class ContinualPPOEWCTrainer(ContinualPPOTrainer):
    """PPO Trainer enhanced with Elastic Weight Consolidation (EWC) for continual learning.

    EWC keeps a memory of parameter importance from previous tasks and penalizes
    changes to important parameters when learning new tasks.
    """

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
    ) -> None:
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

        # EWC-specific attributes
        self.fisher_information: Dict[
            str, torch.Tensor
        ] = {}  # Stores parameter importance
        self.old_params: Dict[
            str, torch.Tensor
        ] = {}  # Stores parameter values after training
        self.task_id: int = 0  # Current task ID
        self.first_task: bool = True  # Flag for the first task

    def compute_loss(
        self, model: Union[PreTrainedModel, nn.Module], inputs: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute PPO loss with additional EWC regularization to prevent catastrophic forgetting."""
        # Regular PPO loss calculation
        regular_loss = super().compute_loss(model, inputs)

        # Skip EWC loss on first task since there's nothing to preserve yet
        if self.first_task:
            return regular_loss

        # Calculate EWC regularization loss
        ewc_loss = self._compute_ewc_loss()

        # Combine losses
        ewc_lambda = getattr(self.args, 'ewc_lambda', 100.0)
        total_loss = regular_loss + ewc_lambda * ewc_loss

        self.log(
            {
                'ewc_loss': ewc_loss.item(),
                'ppo_loss': regular_loss.item(),
                'total_loss': total_loss.item(),
            }
        )

        return total_loss

    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss - penalizes changes to
        parameters that were important for previous tasks.
        """
        ewc_loss = torch.tensor(0.0, device=self.accelerator.device)

        # Get the policy model or equivalent
        if hasattr(self.model, 'base_model'):
            # PEFT model case
            model_to_check = self.model
        elif hasattr(self.model, 'policy'):
            model_to_check = self.model.policy
        elif hasattr(self.model, 'policy_model'):
            model_to_check = self.model.policy_model
        else:
            model_to_check = self.model

        # Calculate the EWC loss for all named parameters with fisher information
        for name, param in model_to_check.named_parameters():
            if name in self.fisher_information:
                # L2 distance between current and old parameters, weighted by importance
                ewc_loss += torch.sum(
                    self.fisher_information[name]
                    * (param - self.old_params[name]).pow(2)
                )

        return ewc_loss

    def train(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Override train method to update Fisher information after training."""
        # Regular training
        result = super().train(*args, **kwargs)

        # After training on this task, update Fisher information
        self._update_fisher_information()

        # Store current parameter values for next task's EWC loss
        if hasattr(self.model, 'base_model'):
            # PEFT model case
            model_to_store = self.model
        elif hasattr(self.model, 'policy'):
            model_to_store = self.model.policy
        elif hasattr(self.model, 'policy_model'):
            model_to_store = self.model.policy_model
        else:
            model_to_store = self.model

        self.old_params = {
            name: param.clone().detach()
            for name, param in model_to_store.named_parameters()
        }

        # Update task counter
        self.task_id += 1
        if self.first_task:
            self.first_task = False

        return result

    def _update_fisher_information(self) -> None:
        """Update Fisher information matrix using the current task's data.
        The Fisher information approximates the importance of each parameter.
        """
        # Get the policy model
        if hasattr(self.model, 'base_model'):
            # PEFT model case
            model_for_fisher = self.model
        elif hasattr(self.model, 'policy'):
            model_for_fisher = self.model.policy
        elif hasattr(self.model, 'policy_model'):
            model_for_fisher = self.model.policy_model
        else:
            model_for_fisher = self.model

        # Initialize a new Fisher information dictionary
        new_fisher = {
            name: torch.zeros_like(param)
            for name, param in model_for_fisher.named_parameters()
        }

        # Number of samples to use for Fisher estimation
        num_samples = min(
            getattr(self.args, 'fisher_estimation_samples', 200),
            len(self.train_dataset),
        )

        # Create a small dataloader for Fisher estimation
        fisher_dataloader = self.get_train_dataloader()

        # Collect samples for Fisher estimation
        sample_count = 0
        for batch in fisher_dataloader:
            if sample_count >= num_samples:
                break

            # Forward pass with gradients
            model_for_fisher.zero_grad()

            # For PPO we need to calculate the full loss since it's complex with multiple components
            loss = self.compute_loss(self.model, batch)
            self.accelerator.backward(loss)

            # Accumulate squared gradients as Fisher information estimate
            for name, param in model_for_fisher.named_parameters():
                if param.grad is not None:
                    new_fisher[name] += param.grad.pow(2).detach() / num_samples

            sample_count += batch['input_ids'].shape[0]

        # Decay old Fisher information and add new information
        if self.first_task:
            # For first task, just use the new Fisher information
            self.fisher_information = new_fisher
        else:
            # For subsequent tasks, apply decay and add new Fisher information
            decay = getattr(self.args, 'ewc_importance_decay', 0.5)
            for name in new_fisher.keys():
                if name in self.fisher_information:
                    self.fisher_information[name] = (
                        decay * self.fisher_information[name]
                        + (1 - decay) * new_fisher[name]
                    )
                else:
                    self.fisher_information[name] = new_fisher[name]

        # Log Fisher information statistics
        total_fisher = sum(
            torch.sum(f).item() for f in self.fisher_information.values()
        )
        self.log({'task_id': self.task_id, 'total_fisher_importance': total_fisher})
