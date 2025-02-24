import functools
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
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
class ContinualDPOArguments(ScriptArguments):
    dataset_name: str = field(
        default='debug',
        metadata={'help': 'The name or path of the continual dataset to use.'},
    )


@dataclass
class ContinualDPOConfig(DPOConfig):
    reward_model_path: str = field(
        default='None',
        metadata={
            'help': 'The name or path to the reward models folder containing all rewards models for continual learning dataset.'
        },
    )
    response_length: int = field(
        default=53,
        metadata={
            'help': 'Length of the response. Borrowed from PPOCOnfig and used only for evaluation.'
        },
    )


class ContinualDPOTrainer(DPOTrainer):
    # Shared accelerator instance across all trainer instances
    shared_accelerator: Optional[Accelerator] = None
    accelerator: Accelerator  # now non-optional after creation

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        eval_policy_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional[dict] = None,
    ):
        if args is None:
            raise ValueError('`args` cannot be None')
        super().__init__(
            model,
            ref_model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processing_class,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
            peft_config,
        )
        # setting a reward model only for evaluation purposes
        self.reward_model = reward_model
        if self.reward_model is not None:
            disable_dropout_in_model(self.reward_model)
        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model,
                args.per_device_train_batch_size,
                args.fp16,
                args.bf16,
            )
        else:
            # Ensure reward_model is a model instance (if given as a str then load it)
            if isinstance(self.reward_model, str):
                from transformers import AutoModelForSequenceClassification

                self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                    self.reward_model
                )
            # Ensure accelerator is ready. It should be set already by DPOTrainer.
            assert self.accelerator is not None, 'Accelerator must be initialized'
            self.reward_model = self.reward_model.to(self.accelerator.device)

        self.eval_policy_dataset = self.preprocess_policy_dataset(eval_policy_dataset)
        if eval_policy_dataset is not None:
            # using the same data_collator as in PPO trainer
            data_collator = DataCollatorWithPadding(self.processing_class)
            self.eval_policy_dataloader = DataLoader(
                self.eval_policy_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=data_collator,
                drop_last=True,
            )  # no need to shuffle eval dataset
            # Ensure accelerator is available
            # TODO remove the check once ruff issues are resolved
            # fmt: off
            assert self.accelerator is not None, (
                'Accelerator must be assigned before prepare()'
            )
            # fmt: on
            self.eval_policy_dataloader = self.accelerator.prepare(
                self.eval_policy_dataloader
            )

    def create_accelerator_and_postprocess(self) -> None:
        # Only initialize a new Accelerator if one does not exist
        if ContinualDPOTrainer.shared_accelerator is None:
            super().create_accelerator_and_postprocess()
            ContinualDPOTrainer.shared_accelerator = self.accelerator
        else:
            # Reuse the shared accelerator
            self.accelerator = ContinualDPOTrainer.shared_accelerator
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

    def preprocess_policy_dataset(self, dataset: Dataset) -> Dataset:
        # using the same mapping function as in PPO trainer
        dataset_text_field = 'prompt'

        def tokenize(element: dict) -> dict[str, list[int]]:
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

        dataset = dataset.map(
            apply_chat_template, fn_kwargs={'tokenizer': self.processing_class}
        )
        # Compute only on main process for faster data processing.
        with PartialState().local_main_process_first():
            dataset = prepare_dataset(dataset, self.processing_class)

        return dataset

    def evaluate_policy(self) -> dict:
        mode = self.model.training
        self.model.eval()
        eval_metrics = defaultdict(list)
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            top_k=None,
            do_sample=False,
        )
        with torch.no_grad():
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
                _, score, _ = get_reward(
                    self.reward_model,
                    postprocessed_query_response,
                    processing_class.pad_token_id,
                    context_length,
                )
                eval_metrics['score'].extend(
                    self.accelerator.gather_for_metrics(score).float().cpu().numpy()
                )
        self.model.train(mode)
        return {'eval_' + k: float(np.mean(v)) for k, v in eval_metrics.items()}

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float` or `None`, *optional*, defaults to `None`):
                Start time of the training.
        """
        train_eval = 'train' if 'loss' in logs else 'eval'
        print(f'Logging {train_eval} metrics...')
        if train_eval == 'eval':
            print('Computing policy metrics...')
            eval_policy_metrics = self.evaluate_policy()
            logs.update(eval_policy_metrics)
        return super().log(logs, start_time)
