from .domain import Domain
from .objective import Objective
from .preference import Preference


class AlignmentTask:
    r"""Encapsulates the specification of an alignment problem.

    Args:
        domain (Domain): Domain of the alignment task.
        objective (Objective): Objective of the alignment task.
        preference (Preference): Preference of the alignment task.
    """

    def __init__(
        self, domain: Domain, objective: Objective, preference: Preference
    ) -> None:
        self._domain = domain
        self._objective = objective
        self._preference = preference

    def __str__(self) -> str:
        return f'Domain: {self.domain}, Objective: {self.objective}, Preference: {self.preference}'

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def objective(self) -> Objective:
        return self._objective

    @property
    def preference(self) -> Preference:
        return self._preference
