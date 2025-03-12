import functools
import inspect
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, PartialState
from datasets import Dataset
from packaging import version
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
    Trainer,
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
    # [No changes needed]
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
    # [No changes needed]
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


# Needed for Zero-3 compatilbility
class COPRState(nn.Module):
    def __init__(self, initial_log_lambda: float):
        super().__init__()
        self.register_buffer(
            'log_lambda', torch.tensor([initial_log_lambda]), persistent=True
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
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
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

        eval_policy_dataset = eval_dataset

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

        # Instead of calling self.register_buffer directly (Trainer is not an nn.Module),
        # create a small module to hold COPR state.
        self._copr_state = COPRState(getattr(args, 'log_lambda', 0.0001))

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

        self.current_task: str = 'task_0'
        self.is_final_eval: bool = False

    def create_accelerator_and_postprocess(self) -> None:
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

    @property
    def lambda_value(self) -> float:
        """Get the current value of the Lagrangian multiplier."""
        return self._copr_state.log_lambda.exp().item()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Compute the COPR loss with alternating Lagrangian multiplier updates."""
        model_output = self.concatenated_forward(model, inputs)

        if 'ref_chosen_logps' in inputs and 'ref_rejected_logps' in inputs:
            ref_chosen_logps = inputs['ref_chosen_logps']
            ref_rejected_logps = inputs['ref_rejected_logps']
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(inputs)

        batch_size = len(model_output['chosen_logps'])
        n_new = batch_size // 2

        chosen_logps = model_output['chosen_logps']
        rejected_logps = model_output['rejected_logps']

        policy_advantage = chosen_logps - rejected_logps

        dpo_loss = -F.logsigmoid(policy_advantage).mean()

        rdpo_loss = (
            (chosen_logps[:n_new] - ref_chosen_logps[:n_new]).pow(2)
            + (rejected_logps[:n_new] - ref_rejected_logps[:n_new]).pow(2)
        ).mean()

        reg_loss = (
            (chosen_logps[n_new:] - ref_chosen_logps[n_new:]).pow(2)
            + (rejected_logps[n_new:] - ref_rejected_logps[n_new:]).pow(2)
        ).mean()

        if reg_loss.item() == 0.0:
            reg_loss = torch.tensor(1e-9, device=reg_loss.device)

        constraint_threshold = getattr(self.args, 'constraint_threshold', 0.1)
        Jc = (reg_loss - constraint_threshold).detach().cpu().item()

        lambda_lr = getattr(self.args, 'lambda_lr', 0.0001)
        with torch.no_grad():
            self._copr_state.log_lambda += lambda_lr * self.lambda_value * Jc

        Lambda = 1 + self.lambda_value
        coef_dpo = getattr(self.args, 'coef_dpo', 0.8)

        loss = (
            coef_dpo * rdpo_loss + self.lambda_value / Lambda * reg_loss + dpo_loss
        ) / Lambda

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

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

    def preprocess_policy_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset for policy evaluation."""
        # [No changes needed]
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
        # [No changes needed]
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

    # [No changes needed for other methods]
    def store_metrics(
        self,
        metrics: Dict,
        train_eval: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        """Override store_metrics to organize metrics by task."""
        if not hasattr(self, '_stored_metrics'):
            self._stored_metrics: Dict = defaultdict(lambda: defaultdict(list))

        # Use train_eval if provided, otherwise fall back to split
        effective_split = train_eval if train_eval is not None else split

        # Add task prefix to eval metrics for better organization
        if effective_split == 'eval' and hasattr(self, 'current_task'):
            # Store metrics both in default location and in task-specific location
            for key, value in metrics.items():
                self._stored_metrics[effective_split][key].append(value)
                task_key = f'{self.current_task}/{key}'
                self._stored_metrics['task'][task_key].append(value)

                # For final evaluation, also store in eval.last
                if self.is_final_eval:
                    self._stored_metrics['eval.last'][key].append(value)
        else:
            # For training metrics, keep original behavior but also add to task
            for key, value in metrics.items():
                self._stored_metrics[effective_split or 'train'][key].append(value)

                if hasattr(self, 'current_task'):
                    task_key = f'{self.current_task}/{key}'
                    self._stored_metrics['task'][task_key].append(value)

    def log(
        self, logs: Dict[str, Union[float, str]], start_time: Optional[float] = None
    ) -> None:
        """Reorganize logs into train, task, and eval.last categories."""
        train_eval = 'train' if 'loss' in logs else 'eval'

        # Process stored metrics
        processed_logs = {}

        # Process the train/eval metrics
        for key, metrics in self._stored_metrics.get(train_eval, {}).items():
            if metrics:
                processed_logs[key] = torch.tensor(metrics).mean().item()

        # Process task-specific metrics
        for key, metrics in self._stored_metrics.get('task', {}).items():
            if metrics:
                processed_logs[f'task/{key}'] = torch.tensor(metrics).mean().item()

        # Process eval.last metrics (final evaluation results)
        if self.is_final_eval and 'eval.last' in self._stored_metrics:
            for key, metrics in self._stored_metrics['eval.last'].items():
                if metrics:
                    processed_logs[f'eval.last/{key}'] = (
                        torch.tensor(metrics).mean().item()
                    )

        # Add lambda value to logs
        processed_logs['lambda'] = self.lambda_value

        # Clear stored metrics after processing
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Update the logs with our processed logs
        logs.update(processed_logs)

        # Call the parent's log method
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            return Trainer.log(self, logs, start_time)
        else:  # transformers<=4.46
            return Trainer.log(self, logs)

    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Override evaluate to mark when we're doing final evaluation."""
        # Check if this is the final evaluation (can be set by external code)
        result = super().evaluate(*args, **kwargs)
        return result

    def set_task(self, task_name: str) -> 'COPRTrainer':
        """Set the current task name for better metric organization."""
        self.current_task = task_name
        return self

    def mark_final_eval(self, is_final: bool = True) -> 'COPRTrainer':
        """Mark that the next evaluation will be the final one for the current task."""
        self.is_final_eval = is_final
        return self
