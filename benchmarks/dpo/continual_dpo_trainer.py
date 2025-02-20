import functools
import inspect
import os
from dataclasses import dataclass, field
from typing import Optional

from accelerate import Accelerator
from trl import DPOTrainer, ScriptArguments


@dataclass
class ContinualDPOArguments(ScriptArguments):
    dataset_name: str = field(
        default='debug',
        metadata={'help': 'The name or path of the continual dataset to use.'},
    )

    wandb_project: Optional[str] = field(
        default=None,
        metadata={'help': 'Override the default WandB project name.'},
    )

    def __post_init__(self) -> None:
        if self.wandb_project:
            os.environ['WANDB_PROJECT'] = self.wandb_project


class ContinualDPOTrainer(DPOTrainer):
    # Shared accelerator instance across all trainer instances
    shared_accelerator: Optional[Accelerator] = None
    accelerator: Optional[Accelerator] = None

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
