from abc import ABC, abstractmethod
from typing import Tuple


class AlignmentDatasetSampleBase(ABC):
    r"""Base class for Alignment Dataset Sample."""

    @abstractmethod
    def prompt(self) -> str:
        r"""Get the prompt associated with this sample."""

    @abstractmethod
    def response(self) -> Tuple[str, str]:
        r"""Get the responses associated with this sample."""

    @abstractmethod
    def winning_response(self) -> str:
        r"""Get the winning response associated with this sample."""

    @abstractmethod
    def losing_response(self) -> str:
        r"""Get the losing response associated with this sample."""
