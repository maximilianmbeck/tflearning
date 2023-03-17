import pickle
from pathlib import Path
from typing import Callable, Type

import numpy as np
import torch
from torch.utils import data


def _get_sample_idxes_by_metric(sample_idx_metric : np.ndarray, fraction: float, best: bool = True) -> np.ndarray:
    """Returns the idxes of the `fraction` of samples with the best or worst metric value."""
    assert sample_idx_metric.ndim == 1, f"sample_idx_metric.ndim must be 1, but is {sample_idx_metric.ndim}"
    sorted_idxes = sample_idx_metric.argsort() # sorts ascending
    if best:
        sorted_idxes = sorted_idxes[::-1] # reverse order, so that best samples are first
    n_samples = int(len(sorted_idxes) * fraction)
    return sorted_idxes[:n_samples]

def _load_dict(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

class RandomSampleSelector:

    def __init__(self, fraction: float):
        self.fraction = fraction

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        n_samples = len(dataset)
        n_samples_to_select = int(n_samples * self.fraction)
        indices = torch.randperm(n_samples)[:n_samples_to_select]
        return data.Subset(dataset, indices)
    
class PredictionDepthSampleSelector:

    def __init__(self, fraction: float, pred_results_file: str, best_samples: bool = True):
        self.fraction = fraction
        self.pred_results_file = Path(pred_results_file)
        self.best_samples = best_samples
        # TODO from here: load the prediction results and get the sample idxes

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

