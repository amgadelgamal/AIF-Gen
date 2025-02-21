import functools
import inspect
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union

from accelerate import Accelerator
from accelerate import PartialState
from trl import DPOTrainer, ScriptArguments
from datasets import Dataset, IterableDataset
from accelerate.utils import broadcast, gather_object
from collections import defaultdict
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    batch_generation,
    get_reward,
    print_rich_table,
    truncate_response,
    disable_dropout_in_model,
    prepare_deepspeed,
)
from trl.trainer.dpo_config import DPOConfig
from trl import apply_chat_template
from transformers import GenerationConfig, DataCollatorWithPadding
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput



@dataclass
class ContinualDPOArguments(ScriptArguments):
    dataset_name: str = field(
        default='debug',
        metadata={'help': 'The name or path of the continual dataset to use.'},
    )

@dataclass
class ContinualDPOConfig(DPOConfig):
    reward_model_path: str = field(
        default=None,
        metadata={'help': 'The name or path to the reward models folder containing all rewards models for continual learning dataset.'},
    )
    response_length: int = field(
        default=53,
        metadata={"help": "Length of the response. Borrowed from PPOCOnfig and used only for evaluation."},
    )


class ContinualDPOTrainer(DPOTrainer):
    # Shared accelerator instance across all trainer instances
    shared_accelerator: Optional[Accelerator] = None
    accelerator: Optional[Accelerator] = None

    def __init__(
            self,
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            reward_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: Optional[DPOConfig] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
            processing_class: Optional[Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            peft_config: Optional[dict] = None,
    ):
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
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
        else:
            self.reward_model = self.reward_model.to(self.accelerator.device)

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

    def evaluate_policy(self, dataset) -> dict:
        """
        Evaluate the model on the evaluation dataset and compute metrics.
        """
        # ToDo: the whole dataset preparation stage inside evaluate_policy funciton doesn't look pretty, probably should be moved to a separate method
        # using the same mapping function as in PPO trainer
        dataset_text_field = "prompt"

        def prepare_dataset(dataset, tokenizer):
            """pre-tokenize the dataset before training; only collate during training"""

            def tokenize(element):
                outputs = tokenizer(
                    element[dataset_text_field],
                    padding=False,
                )
                return {"input_ids": outputs["input_ids"]}

            return dataset.map(
                tokenize,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=self.args.dataset_num_proc,
            )

        dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": self.processing_class})
        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            dataset = prepare_dataset(dataset, self.processing_class)
        # using the same data_collator as in PPO trainer
        data_collator = DataCollatorWithPadding(self.processing_class)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        dataloader = self.accelerator.prepare(dataloader)

        mode = self.model.training
        self.model.eval()
        eval_metrics = defaultdict(list)
        processing_class = self.processing_class
        # ToDo: we sample response greedily here, but is it the best way to do it for evaluation?
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            # temperature=0.01,
            top_k=None,
            # top_p=1.0,
            do_sample=False,
        )

        with torch.no_grad():
            for batch in dataloader:
                query = batch["input_ids"].to(self.accelerator.device)
                context_length = query.shape[1]

                with unwrap_model_for_generation(
                        self.model, self.accelerator, gather_deepspeed3_params=None #self.args.ds3_gather_for_generation
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
                # ToDo: stop_token_id is None for default PPO config, check if we need it anyway
                # if self.stop_token_id is not None:
                #     postprocessed_response = truncate_response(
                #         self.stop_token_id, processing_class.pad_token_id, response
                #     )

                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                _, score, _ = get_reward(
                    self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                )
                eval_metrics["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

        eval_metrics["score"] = np.mean(eval_metrics["score"])
        self.model.train(mode)

        return eval_metrics


