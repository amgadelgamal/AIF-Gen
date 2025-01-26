import sys
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional

from benchmarks.COPR.trlx.data.configs import TRLConfig
from benchmarks.COPR.trlx.pipeline import BaseRolloutStore

# specifies a dictionary of architectures
_TRAINERS: Dict[str, Any] = {}  # registry


def register_trainer(cls):
    """Register a trainer class.

    Args:
        cls: The trainer class to register.
    """
    name = cls.__name__.lower()
    _TRAINERS[name] = cls
    setattr(sys.modules[__name__], name, cls)
    print(f'Registered trainer: {name}')  # Debug print
    return cls


@register_trainer
class BaseRLTrainer:
    def __init__(
        self,
        config: TRLConfig,
        reward_fn=None,
        metric_fn=None,
        logit_mask=None,
        stop_sequences=None,
        train_mode=False,
    ):
        self.store: BaseRolloutStore = None
        self.config = config
        self.reward_fn = reward_fn
        self.metric_fn = metric_fn
        self.train_mode = train_mode
        self.logit_mask = logit_mask
        self.stop_sequences = stop_sequences

    def push_to_store(self, data):
        self.store.push(data)

    def add_eval_pipeline(self, eval_pipeline):
        """Adds pipeline for validation prompts"""
        self.eval_pipeline = eval_pipeline

    @abstractmethod
    def sample(
        self, prompts: Iterable[str], length: int, n_samples: int
    ) -> Iterable[str]:
        """Sample from the language. Takes prompts and maximum length to generate.

        :param prompts: List of prompts to tokenize and use as context

        :param length: How many new tokens to genrate for each prompt
        :type length: int

        :param n_samples: Default behavior is to take number of prompts as this
        """

    @abstractmethod
    def learn(
        self,
        log_fn: Callable = None,
        save_fn: Callable = None,
        eval_fn: Callable = None,
    ):
        """Use experiences in RolloutStore to learn

        :param log_fn: Optional function that is called when logging and passed a dict of logging relevant values
        :type log_fn: Callable[Dict[str, any]]

        :param save_fn: Optional function to call after saving. Is passed the components.
        :type save_fn: Callable[Dict[str, any]]

        :param eval_fn: Optional function to call during evaluation. Eval doesn't do anything without this.
        :type eval_fn: Callable[BaseRLTrainer]
        """

    @abstractmethod
    def save(self, directory: Optional[str] = None):
        """Creates a checkpoint of training states"""

    @abstractmethod
    def load(self, directory=None):
        """Loads a checkpoint created from `save`"""
