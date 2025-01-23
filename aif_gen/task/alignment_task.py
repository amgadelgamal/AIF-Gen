from typing import Any, Dict

from .domain import Domain


class AlignmentTask:
    r"""Encapsulates the specification of an alignment problem.

    Args:
        domain (Domain): Domain of the alignment task.
        objective (str): Description of the alignment objective.
        preference (str): Description of the objective preference.
    """

    def __init__(self, domain: Domain, objective: str, preference: str) -> None:
        self._domain = domain
        self._objective = objective
        self._preference = preference

    @classmethod
    def from_dict(cls, task_dict: Dict[str, Any]) -> 'AlignmentTask':
        expected_keys = {'domain', 'objective', 'preference'}
        missing_keys = expected_keys - set(task_dict)
        if len(missing_keys):
            raise ValueError(f'Missing required keys: {missing_keys}')

        domain = Domain.from_dict(task_dict['domain'])
        objective = task_dict['objective']
        preference = task_dict['preference']
        return cls(domain, objective, preference)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'domain': self.domain.to_dict(),
            'objective': self.objective,
            'preference': self.preference,
        }

    def __str__(self) -> str:
        return f'AlignmentTask({self.domain}, Objective: {self.objective}, Preference: {self.preference})'

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def objective(self) -> str:
        return self._objective

    @property
    def preference(self) -> str:
        return self._preference
