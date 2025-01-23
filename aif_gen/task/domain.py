from typing import Any, Dict


class Domain:
    def __init__(self, domain: str) -> None:
        self._domain = domain

    @classmethod
    def from_dict(cls, domain_dict: Dict[str, Any]) -> 'Domain':
        return cls('Mock Domain')

    def to_dict(self) -> Dict[str, str]:
        return {'Mock Domain': 'Mock Domain Value'}

    def __str__(self) -> str:
        return f'Domain: {self._domain}'

    @property
    def domain(self) -> str:
        return self._domain
