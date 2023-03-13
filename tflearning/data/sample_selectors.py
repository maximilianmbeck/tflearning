import torch
from typing import Type, Callable
from torch.utils import data


class RandomSampleSelector:

    def __init__(self, fraction: float):
        self.fraction = fraction

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        n_samples = len(dataset)
        n_samples_to_select = int(n_samples * self.fraction)
        indices = torch.randperm(n_samples)[:n_samples_to_select]
        return data.Subset(dataset, indices)

_sample_selector_registry = {
    'random': RandomSampleSelector,
}


def get_sample_selector_class(sample_selector_name: str) -> Type[Callable]:
    if sample_selector_name in _sample_selector_registry:
        return _sample_selector_registry[sample_selector_name]
    else:
        assert False, f"Unknown sample selector name \"{sample_selector_name}\". Available sample selectors are: {str(_sample_selector_registry.keys())}"

