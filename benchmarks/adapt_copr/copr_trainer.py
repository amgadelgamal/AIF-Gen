import functools
import inspect
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollator,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer, ScriptArguments, apply_chat_template
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.utils import (
    batch_generation,
    disable_dropout_in_model,
    get_reward,
    prepare_deepspeed,
)


@dataclass
class COPRArguments(ScriptArguments):
    dataset_name: str = field(
        default='debug',
        metadata={'help': 'The name or path of the continual dataset to use.'},
    )
    wandb_project: Optional[str] = field(
        default='AIFGen-copr-continual-test',
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
class COPRConfig(DPOConfig):
    """Configuration class for COPRTrainer with COPR-specific parameters."""

    log_lambda: float = field(
        default=0.0001,
        metadata={'help': 'Initial value for the log of Lagrangian multiplier'},
    )
    lambda_lr: float = field(
        default=0.0001, metadata={'help': 'Learning rate for Lagrangian multiplier'}
    )
    constraint_threshold: float = field(
        default=0.1, metadata={'help': 'Threshold for constraint violation'}
    )
    coef_dpo: float = field(
        default=0.8, metadata={'help': 'Coefficient for DPO loss term'}
    )
    memory_buffer_size: int = field(
        default=100, metadata={'help': 'Number of examples to keep in memory buffer'}
    )
    reward_model_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The name or path to the reward models folder containing all rewards models for continual learning dataset.'
        },
    )
    mock: bool = field(
        default=False,
        metadata={'help': 'Whether to use mock dataset.'},
    )
    response_length: int = field(
        default=53,
        metadata={'help': 'Length of the response. Used only for evaluation.'},
    )
    temperature: float = field(
        default=0.7,
        metadata={'help': 'Temperature for sampling. Used only for evaluation.'},
    )
    eval_greedy_policy: bool = field(
        default=False,
        metadata={'help': 'Whether to use greedy policy for evaluation.'},
    )
    buffer_ratio: float = field(
        default=0.1,
        metadata={
            'help': 'Percentage of task samples to keep in memory buffer (0.0-1.0). If specified, overrides memory_buffer_size.'
        },
    )
    use_buffer_ratio: bool = field(
        default=False,
        metadata={'help': 'Whether to use buffer_ratio instead of memory_buffer_size'},
    )


class COPRTrainer(DPOTrainer):
    """COPR Trainer that implements Continual Human Preference Learning via Optimal Policy Regularization."""

    shared_accelerator: Optional[Accelerator] = None
    accelerator: Accelerator

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[COPRConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[
            Dataset
        ] = None,  # This will already contain the mixture of current task + buffer
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        eval_policy_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional[Dict] = None,
    ):
        if args is None:
            raise ValueError('`args` cannot be None')

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        # Initialize COPR-specific attributes
        self.log_lambda = torch.tensor(
            [getattr(args, 'log_lambda', 0.0001)], device=self.accelerator.device
        )

        # Set up reward model for evaluation
        self.reward_model = reward_model
        if self.reward_model is not None:
            disable_dropout_in_model(self.reward_model)

        # Configure reward model with proper device settings
        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model,
                    args.per_device_train_batch_size,
                    args.fp16,
                    args.bf16,
                )
        else:
            if isinstance(self.reward_model, str):
                from transformers import AutoModelForSequenceClassification

                self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                    self.reward_model
                )
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)

        # Setup evaluation policy dataset
        if eval_policy_dataset is not None:
            self.eval_policy_dataset = self.preprocess_policy_dataset(
                eval_policy_dataset
            )
            data_collator = DataCollatorWithPadding(self.processing_class)
            self.eval_policy_dataloader = DataLoader(
                self.eval_policy_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=data_collator,
                drop_last=True,
            )
            self.eval_policy_dataloader = self.accelerator.prepare(
                self.eval_policy_dataloader
            )
        else:
            self.eval_policy_dataset = None
            self.eval_policy_dataloader = None

        print(self.ref_model)

    @property
    def lambda_value(self) -> float:
        """Get the current value of the Lagrangian multiplier."""
        return self.log_lambda.exp().item()

    def create_accelerator_and_postprocess(self) -> None:
        """Override to reuse accelerator instance across trainers."""
        # Only initialize a new Accelerator if one does not exist
        if COPRTrainer.shared_accelerator is None:
            super().create_accelerator_and_postprocess()
            COPRTrainer.shared_accelerator = self.accelerator
        else:
            # Reuse the shared accelerator
            self.accelerator = COPRTrainer.shared_accelerator
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

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Compute the COPR loss with alternating Lagrangian multiplier updates."""
        # Process inputs - already tokenized by the time they get here
        chosen_ids = inputs['chosen_input_ids']
        rejected_ids = inputs['rejected_input_ids']

        # Forward pass for chosen responses
        chosen_logits = model(
            chosen_ids, attention_mask=inputs['chosen_attention_mask']
        ).logits

        # Forward pass for rejected responses
        rejected_logits = model(
            rejected_ids, attention_mask=inputs['rejected_attention_mask']
        ).logits

        # Get log probabilities
        chosen_logprobs = F.log_softmax(chosen_logits[:, :-1], dim=-1)
        rejected_logprobs = F.log_softmax(rejected_logits[:, :-1], dim=-1)

        # Get reference log probabilities
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_chosen_logits = self.model(
                        chosen_ids, attention_mask=inputs['chosen_attention_mask']
                    ).logits
                    ref_rejected_logits = self.model(
                        rejected_ids, attention_mask=inputs['rejected_attention_mask']
                    ).logits
            else:
                ref_chosen_logits = self.ref_model(
                    chosen_ids, attention_mask=inputs['chosen_attention_mask']
                ).logits
                ref_rejected_logits = self.ref_model(
                    rejected_ids, attention_mask=inputs['rejected_attention_mask']
                ).logits

            ref_chosen_logprobs = F.log_softmax(ref_chosen_logits[:, :-1], dim=-1)
            ref_rejected_logprobs = F.log_softmax(ref_rejected_logits[:, :-1], dim=-1)

        batch_size = chosen_ids.shape[0]
        n_new = batch_size // 2  # First half are new task, second half are from buffer

        # Extract response token indices
        chosen_response_indices = inputs.get('chosen_response_start_indices', None)
        rejected_response_indices = inputs.get('rejected_response_start_indices', None)

        # If indices not provided, estimate them (this is a simplification)
        if chosen_response_indices is None:
            # Assume prompt length is consistent per batch and is the first part of the sequence
            prompt_length = inputs.get('prompt_length', chosen_ids.shape[1] // 2)
            chosen_response_indices = (
                torch.ones(batch_size, device=chosen_ids.device) * prompt_length
            )
            rejected_response_indices = (
                torch.ones(batch_size, device=rejected_ids.device) * prompt_length
            )

        # Extract token probabilities for responses
        chosen_token_logprobs = []
        ref_chosen_token_logprobs = []
        rejected_token_logprobs = []
        ref_rejected_token_logprobs = []

        for i in range(batch_size):
            start_idx = int(chosen_response_indices[i].item())
            chosen_token_logprobs.append(
                self._get_token_logprobs(
                    chosen_logprobs[i], chosen_ids[i, 1:], start_idx
                )
            )
            ref_chosen_token_logprobs.append(
                self._get_token_logprobs(
                    ref_chosen_logprobs[i], chosen_ids[i, 1:], start_idx
                )
            )

            start_idx = int(rejected_response_indices[i].item())
            rejected_token_logprobs.append(
                self._get_token_logprobs(
                    rejected_logprobs[i], rejected_ids[i, 1:], start_idx
                )
            )
            ref_rejected_token_logprobs.append(
                self._get_token_logprobs(
                    ref_rejected_logprobs[i], rejected_ids[i, 1:], start_idx
                )
            )

        chosen_token_logprobs_tensor = torch.cat(chosen_token_logprobs)
        ref_chosen_token_logprobs_tensor = torch.cat(ref_chosen_token_logprobs)
        rejected_token_logprobs_tensor = torch.cat(rejected_token_logprobs)
        ref_rejected_token_logprobs_tensor = torch.cat(ref_rejected_token_logprobs)

        # Calculate advantage (chosen - rejected)
        policy_advantage = chosen_token_logprobs_tensor - rejected_token_logprobs_tensor

        # Compute DPO-style loss term
        dpo_loss = -F.logsigmoid(policy_advantage).mean()

        # RDPO loss for new task samples (optimization objective)
        rdpo_loss = (
            (
                chosen_token_logprobs_tensor[:n_new]
                - ref_chosen_token_logprobs_tensor[:n_new]
            ).pow(2)
            + (
                rejected_token_logprobs_tensor[:n_new]
                - ref_rejected_token_logprobs_tensor[:n_new]
            ).pow(2)
        ).mean()

        # Regularization loss for old task samples (constraint)
        reg_loss = (
            (
                chosen_token_logprobs_tensor[n_new:]
                - ref_chosen_token_logprobs_tensor[n_new:]
            ).pow(2)
            + (
                rejected_token_logprobs_tensor[n_new:]
                - ref_rejected_token_logprobs_tensor[n_new:]
            ).pow(2)
        ).mean()

        # Ensure regularization loss is non-zero
        if reg_loss.item() == 0.0:
            reg_loss = torch.tensor(1e-9, device=reg_loss.device)

        # Compute constraint violation
        getattr(self.args, 'beta', 0.1)
        constraint_threshold = getattr(self.args, 'constraint_threshold', 0.1)
        Jc = (reg_loss - constraint_threshold).detach().cpu().item()

        # Update Lagrangian multiplier
        lambda_lr = getattr(self.args, 'lambda_lr', 0.0001)
        self.log_lambda = self.log_lambda + lambda_lr * self.lambda_value * Jc

        # Compute final loss with Lagrangian
        Lambda = 1 + self.lambda_value  # Normalization term
        coef_dpo = getattr(self.args, 'coef_dpo', 0.8)

        loss = (
            coef_dpo * rdpo_loss + self.lambda_value / Lambda * reg_loss + dpo_loss
        ) / Lambda

        # Handle device synchronization if needed
        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        # Prepare metrics for logging
        metrics = {
            'loss': loss.item(),
            'dpo_loss': dpo_loss.item(),
            'rdpo_loss': rdpo_loss.item(),
            'reg_loss': reg_loss.item(),
            'lambda': self.lambda_value,
        }

        self.store_metrics(metrics, 'train')

        if return_outputs:
            return loss, {'metrics': metrics}
        return loss

    def _get_token_logprobs(
        self, logprobs: torch.Tensor, tokens: torch.Tensor, response_start_idx: int
    ) -> torch.Tensor:
        """Helper to extract token log probabilities for the response."""
        # Extract only response tokens starting from response_start_idx
        response_logprobs = logprobs[response_start_idx:]
        response_tokens = tokens[response_start_idx:]

        if len(response_tokens) == 0:
            # Handle edge case with no response tokens
            return torch.tensor([0.0], device=logprobs.device)

        # Get log prob for each token
        token_logprobs = torch.gather(
            response_logprobs, dim=-1, index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Return mean log prob
        return token_logprobs.mean().unsqueeze(0)

    def preprocess_policy_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset for policy evaluation."""
        dataset_text_field = 'prompt'

        def tokenize(element: Dict) -> Dict[str, List[int]]:
            outputs = self.processing_class(
                element[dataset_text_field],
                padding=False,
            )
            return {'input_ids': outputs['input_ids']}

        def prepare_dataset(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
            return ds.map(
                tokenize,
                batched=True,
                remove_columns=ds.column_names,
                num_proc=self.args.dataset_num_proc,
            )

        dataset = (
            dataset.map(
                apply_chat_template, fn_kwargs={'tokenizer': self.processing_class}
            )
            if getattr(self.args, 'mock', False)
            else dataset
        )

        # Compute only on main process for faster data processing.
        with PartialState().local_main_process_first():
            dataset = prepare_dataset(dataset, self.processing_class)
        return dataset

    def evaluate_policy(self) -> Dict:
        """Evaluate the policy using the evaluation policy dataloader."""
        mode = self.model.training
        self.model.eval()
        eval_metrics = defaultdict(list)
        processing_class = self.processing_class

        # Configure generation parameters
        if getattr(self.args, 'eval_greedy_policy', False):
            generation_config = GenerationConfig(
                max_new_tokens=getattr(self.args, 'response_length', 53),
                top_k=None,
                do_sample=False,
            )
        else:
            # Using the same hyperparams as during training
            generation_config = GenerationConfig(
                max_new_tokens=getattr(self.args, 'response_length', 53),
                temperature=(getattr(self.args, 'temperature', 0.7) + 1e-7),
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
            )

        with torch.no_grad():
            if self.eval_policy_dataloader is not None:
                for batch in self.eval_policy_dataloader:
                    query = batch['input_ids'].to(self.accelerator.device)
                    context_length = query.shape[1]
                    with unwrap_model_for_generation(
                        self.model,
                        self.accelerator,
                        gather_deepspeed3_params=None,
                    ) as unwrapped_model:
                        query_response, _ = batch_generation(
                            unwrapped_model,
                            query,
                            query.shape[0],
                            processing_class.pad_token_id,
                            generation_config,
                        )
                        response = query_response[:, context_length:]

                    postprocessed_response = response
                    postprocessed_query_response = torch.cat(
                        (query, postprocessed_response), 1
                    )

                    if self.reward_model is not None:
                        _, score, _ = get_reward(
                            self.reward_model,
                            postprocessed_query_response,
                            processing_class.pad_token_id,
                            context_length,
                        )
                        eval_metrics['score'].extend(
                            self.accelerator.gather_for_metrics(score)
                            .float()
                            .cpu()
                            .numpy()
                        )

        self.model.train(mode)
        return {'eval_' + k: float(np.mean(v)) for k, v in eval_metrics.items()}

    def log(
        self, logs: Dict[str, Union[float, str]], start_time: Optional[float] = None
    ) -> None:
        """Add COPR-specific metrics to logs including policy evaluation metrics."""
        train_eval = 'train' if 'loss' in logs else 'eval'
        print(f'Logging {train_eval} metrics...')

        if train_eval == 'eval' and self.eval_policy_dataloader is not None:
            print('Computing policy metrics...')
            eval_policy_metrics = self.evaluate_policy()
            logs.update(eval_policy_metrics)

        # Add lambda value to logs
        logs['lambda'] = self.lambda_value

        return super().log(logs, start_time)
