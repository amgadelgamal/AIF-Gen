from typing import Any, Dict, List, Optional

from .domain_component import DomainComponent


class Domain:
    r"""A domain is a combination of DomainComponents along with a weight for each component.

    Args:
        components (List[DomainComponent]): List of DomainComponents that constitute the domain.
        weights (Optional[List[float]]): Weights given to each constituent component (uniform if not specified).
    """

    def __init__(
        self, components: List[DomainComponent], weights: Optional[List[float]] = None
    ) -> None:
        if not len(components):
            raise ValueError(
                'Cannot initialize a Domain with an empty list of DomainComponents'
            )

        if weights is None:
            weights = [1 / len(components)] * len(components)

        if len(weights) != len(components):
            raise ValueError(
                f'Number of components and weights must match, but got {len(components)} components and {len(weights)} weights'
            )
        for i, weight in enumerate(weights):
            if weight < 0:
                raise ValueError(
                    f'Got a negative weight for component: {components[i]}'
                )

        self._components = components
        self._weights = weights

    @classmethod
    def from_dict(cls, domain_dict: Dict[str, Dict[str, Any]]) -> 'Domain':
        components, component_weights = [], []
        for component_name, component_args in domain_dict.items():
            component_args['name'] = component_name
            components.append(DomainComponent.from_dict(component_args))

            if 'weight' in component_args:
                component_weights.append(component_args['weight'])

        weights = None if not len(component_weights) else component_weights
        return cls(components, weights)

    def __str__(self) -> str:
        s = f'Domain: ['
        for i in range(self.num_components):
            s += f'({self.components[i]}, weight={self.weights[i]:.2f}), '
        s += ']'
        return s

    @property
    def components(self) -> List[DomainComponent]:
        return self._components

    @property
    def num_components(self) -> int:
        return len(self.components)

    @property
    def weights(self) -> List[float]:
        return self._weights
