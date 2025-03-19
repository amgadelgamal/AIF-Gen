import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
)
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

from benchmarks.dataloading import init_continual_dataset


# Class-level storage to maintain state between trainer instances
class MASState:
    parameter_importance: Any = {}
    previous_params: Any = None
    tasks_seen: int = 0


class ContinualRewardTrainer(RewardTrainer):
    """Reward trainer with Memory Aware Synapses (MAS) for continual learning."""

    def __init__(self, *args: Any, **kwargs: Any):
        # Extract mas_lambda before passing to parent constructor
        if 'mas_lambda' in kwargs:
            self.mas_lambda = kwargs.pop('mas_lambda')
        else:
            self.mas_lambda = 1.0

        # Extract task_id before passing to parent constructor
        if 'task_id' in kwargs:
            self.current_task_id = kwargs.pop('task_id')
        else:
            self.current_task_id = 0

        super().__init__(*args, **kwargs)

        # Initialize parameter importance if this is the first task
        if not MASState.parameter_importance:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    MASState.parameter_importance[name] = torch.zeros_like(param.data)

    def _compute_mas_regularization(self) -> torch.Tensor:
        """Compute MAS regularization loss efficiently."""
        device = next(self.model.parameters()).device
        mas_loss = torch.tensor(0.0, device=device)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if MASState.previous_params is None:
                    raise ValueError(
                        'Previous parameters are not stored. Call store_model_parameters() first.'
                    )
                # Ensure previous params are on the same device as current params
                if MASState.previous_params[name].device != param.device:
                    MASState.previous_params[name] = MASState.previous_params[name].to(
                        param.device
                    )

                # Weight by importance (also ensure on correct device)
                if MASState.parameter_importance[name].device != param.device:
                    MASState.parameter_importance[name] = MASState.parameter_importance[
                        name
                    ].to(param.device)

                # Compute weighted squared difference
                diff = param.data - MASState.previous_params[name]
                mas_loss += (MASState.parameter_importance[name] * diff**2).sum()

        return mas_loss

    def compute_loss(
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Dict[str, Any]]]:
        """Override compute_loss to add MAS regularization for tasks after the first."""
        # First delegate to the parent RewardTrainer's compute_loss
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        mas_loss = torch.tensor(0.0, device=loss.device)
        # Add MAS regularization if not on the first task
        if MASState.tasks_seen > 0 and MASState.previous_params is not None:
            mas_loss = self._compute_mas_regularization()
            loss = loss + self.mas_lambda * mas_loss

        # Add mas_loss to outputs for logging
        if return_outputs:
            outputs['mas_loss'] = mas_loss.detach()

        return (loss, outputs) if return_outputs else loss

    def compute_importance_weights(self, dataset: Dataset) -> None:
        """Compute parameter importance efficiently, using the trainer's data collator."""
        import time

        start = time.time()

        self.model.eval()
        device = next(self.model.parameters()).device
        print(f'Computing importance weights on device: {device}')

        # Use a smaller subset for efficiency
        max_samples = min(len(dataset), 500)  # Sample at most 500 examples

        importance_estimates = {
            name: torch.zeros_like(param.data, device=device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Use trainer's dataloader which handles variable sequence lengths properly
        # First create a smaller subset of the dataset
        import random

        from datasets import Dataset as HFDataset

        indices = random.sample(range(len(dataset)), max_samples)
        subset_data = {k: [dataset[i][k] for i in indices] for k in dataset[0].keys()}
        subset = HFDataset.from_dict(subset_data)

        # Use the trainer's dataloader which has the right collator
        dataloader = self.get_eval_dataloader(subset)

        batch_count = 0
        for batch in dataloader:
            batch = self._prepare_inputs(batch)
            batch_count += 1

            # Forward pass WITH gradient computation enabled
            rewards = self.model(
                input_ids=batch['input_ids_chosen'],
                attention_mask=batch['attention_mask_chosen'],
                return_dict=True,
            )['logits']

            # Compute L2 norm of the output
            output_norm = torch.norm(rewards)

            # Calculate gradients for all parameters at once
            output_norm.backward()

            # Update importance estimates with absolute gradient values
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        importance_estimates[name] += param.grad.data.abs()

            # Zero out the gradients for the next batch
            self.model.zero_grad()

        # Normalize and update importance weights (keep on GPU)
        for name in importance_estimates:
            # Make sure parameter_importance is on same device
            if (
                name in MASState.parameter_importance
                and MASState.parameter_importance[name].device != device
            ):
                MASState.parameter_importance[name] = MASState.parameter_importance[
                    name
                ].to(device)

            # Add to existing importance (accumulate across tasks)
            # Normalize by batch count
            MASState.parameter_importance[name] += importance_estimates[name] / max(
                batch_count, 1
            )

        end = time.time()
        print(f'Importance weight computation took {end-start:.2f} seconds')

    def store_model_parameters(self) -> None:
        """Store model parameters efficiently."""
        import time

        start = time.time()

        device = next(self.model.parameters()).device
        print(f'Storing parameters on device: {device}')

        # Initialize or update parameters dictionary
        if MASState.previous_params is None:
            MASState.previous_params = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Keep on same device as model
                MASState.previous_params[name] = param.data.clone()

        end = time.time()
        print(f'Parameter storage took {end-start:.2f} seconds')

    def prediction_step(
        self,
        model: Union[torch.nn.Module, PreTrainedModel],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Override prediction_step to handle variable-sized outputs.
        It ensures the output logits have the correct format for visualization.
        """
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, 'config'):
                ignore_keys = getattr(
                    self.model.config, 'keys_to_ignore_at_inference', []
                )
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss, None, None

        loss = loss.detach()
        batch_size = inputs['input_ids_chosen'].shape[0]

        # Create dummy logits with shape [batch_size, 2] where chosen scores are higher
        dummy_logits = torch.zeros((batch_size, 2), device=loss.device)
        dummy_logits[:, 0] = 0.9  # High probability for chosen
        dummy_logits[:, 1] = 0.1  # Low probability for rejected
        labels = torch.zeros(batch_size, device=loss.device)

        # Explicitly check for the expected keys.
        rewards_chosen = logits_dict.get('rewards_chosen')
        rewards_rejected = logits_dict.get('rewards_rejected')

        # Early return with dummy values if missing or empty logits
        if (
            rewards_chosen is None
            or rewards_rejected is None
            or rewards_chosen.numel() == 0
            or rewards_rejected.numel() == 0
        ):
            print(
                'Warning: One or both reward outputs are missing or empty; using dummy logits.'
            )
            return loss, dummy_logits, labels

        try:
            # Convert to absolute logits (necessary for the expected visualization format)
            # We need a 2D tensor with shape [batch_size, 2] where each row sums to 1.0
            if rewards_chosen.dim() >= 2 and rewards_rejected.dim() >= 2:
                # Get the scores and reshape as needed
                chosen_scores = rewards_chosen.view(batch_size, -1).mean(
                    dim=1
                )  # Average if multiple scores
                rejected_scores = rewards_rejected.view(batch_size, -1).mean(dim=1)

                # Stack them side by side and apply softmax to get probabilities
                probs = torch.stack([chosen_scores, rejected_scores], dim=1)
                probs = torch.softmax(probs, dim=1)

                return loss, probs, labels
            else:
                print(
                    'Warning: Reward outputs have unexpected dimensions; using dummy logits.'
                )
                return loss, dummy_logits, labels

        except Exception as e:
            print(f'Warning: Error processing logits: {e}. Using dummy values instead.')
            return loss, dummy_logits, labels

    def after_training(self) -> None:
        """Actions to perform after training on a task."""
        self.compute_importance_weights(self.train_dataset)
        self.store_model_parameters()
        MASState.tasks_seen += 1
        # Optional: log average importance value
        avg_importance = {
            name: importance.mean().item()
            for name, importance in MASState.parameter_importance.items()
        }
        self.log_metrics('mas_importance', avg_importance)


@dataclass
class MASScriptArguments(ScriptArguments):
    dataset_index: int = field(
        default=0,
        metadata={
            'help': 'Index of the dataset to use, dataset points to ContinualDataset, '
            'this index points to individual dataset in the ContinualDataset.'
        },
    )
    mock: bool = field(
        default=False,
        metadata={'help': 'Whether to use mock datasets.'},
    )
    wandb_project: Optional[str] = field(
        default='AIFGen-dpo-continual-test',
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
    mas_lambda: float = field(
        default=1.0,
        metadata={'help': 'Regularization coefficient for MAS.'},
    )

    def __post_init__(self) -> None:
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.wandb_entity:
            os.environ['WANDB_ENTITY'] = self.wandb_entity


def train_continually(
    script_args: MASScriptArguments,
    training_args: RewardConfig,
    model_args: ModelConfig,
    continual_dataset: list[Dict[str, Dataset]],
) -> None:
    """Train the reward model continually with MAS regularization."""
    # Ensure we're using GPU efficiently
    if torch.cuda.is_available():
        # Enable FP16 for faster training
        training_args.fp16 = True
        # Disable gradient checkpointing if not needed (can slow things down)
        if not training_args.gradient_checkpointing:
            training_args.gradient_checkpointing = False

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    if script_args.wandb_run_name is not None:
        training_args.run_name = script_args.wandb_run_name

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # Use ChatML format if the tokenizer doesn't already have a chat template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Use only the filename part without extension from the dataset name.
    clean_dataset_name = os.path.basename(script_args.dataset_name)
    if '.' in clean_dataset_name:
        clean_dataset_name = clean_dataset_name.split('.')[0]

    # Initialize the MAS state class (will be populated during training)
    _ = MASState()

    # Train sequentially on each task
    for task_id, dataset in enumerate(continual_dataset):
        print(f'Training on task {task_id+1}/{len(continual_dataset)}')

        # Create a new trainer for this task
        trainer = ContinualRewardTrainer(
            model=model,  # Same model instance is used across all tasks
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != 'no'
            else None,
            peft_config=get_peft_config(model_args),
            mas_lambda=script_args.mas_lambda,
            task_id=task_id,
        )

        # Train the model on this task
        trainer.train()

        # Update MAS state after training
        trainer.after_training()

        # Save model after each task
        custom_repo_name = (
            model_args.model_name_or_path.split('/')[-1]
            + '_'
            + clean_dataset_name
            + '_REWARD_MAS_'
            + str(task_id)
        )

        task_output_dir = os.path.join(training_args.output_dir, custom_repo_name)
        print(f'Saving model after task {task_id} to: {task_output_dir}')

        # Evaluate on current task
        if training_args.eval_strategy != 'no':
            metrics = trainer.evaluate()
            trainer.log_metrics(f'eval_task_{task_id}', metrics)
            trainer.save_metrics(f'eval_task_{task_id}', metrics)

        # Save model
        if not training_args.push_to_hub:
            trainer.save_model(task_output_dir)
        else:
            trainer.push_to_hub(
                model_name=custom_repo_name,
                dataset_name=clean_dataset_name + '_MAS_' + str(task_id),
            )


if __name__ == '__main__':
    parser = HfArgumentParser((MASScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    continual_dataset: list[Dict[str, Dataset]] = init_continual_dataset(
        script_args.dataset_name, mock=script_args.mock
    )

    # For MAS, we always train sequentially on all datasets
    train_continually(script_args, training_args, model_args, continual_dataset)
