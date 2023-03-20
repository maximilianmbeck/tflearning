import pickle
from pathlib import Path
from typing import Callable, Type

import numpy as np
import torch
from torch.utils import data


def _get_sample_idxes_by_metric(sample_idx_metric: np.ndarray,
                                fraction: float,
                                keep_highest: bool = True) -> np.ndarray:
    """Returns the idxes of the `fraction` of samples with the highest or lowest metric value."""
    assert sample_idx_metric.ndim == 1, f"sample_idx_metric.ndim must be 1, but is {sample_idx_metric.ndim}"
    sorted_idxes = sample_idx_metric.argsort()  # sorts ascending
    if keep_highest:
        sorted_idxes = sorted_idxes[::-1]  # reverse order, so that best samples are first
    n_samples = int(len(sorted_idxes) * fraction)
    return sorted_idxes[:n_samples]


def _load_dict(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_random_subset_idxes(indices: np.ndarray, restrict_n_samples: int = -1) -> np.ndarray:
    """Take an iid subset of the indices."""
    if restrict_n_samples > 0:
        assert restrict_n_samples <= len(
            indices
        ), f"Not enough samples in the datset: restrict_n_samples ({restrict_n_samples}) must be smaller than len(indices) ({len(indices)}). "
        indices = indices[torch.randperm(len(indices))[:restrict_n_samples]]
    return indices


class RandomSampleSelector:

    def __init__(self, fraction: float, restrict_n_samples: int = -1):
        self.fraction = fraction
        self.restrict_n_samples = restrict_n_samples

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        n_samples = len(dataset)
        n_samples_to_select = int(n_samples * self.fraction)
        indices = torch.randperm(n_samples)[:n_samples_to_select]
        indices = get_random_subset_idxes(indices, self.restrict_n_samples)
        return data.Subset(dataset, indices)


class PredictionDepthSampleSelector:

    def __init__(self,
                 fraction: float,
                 pred_results_file: str,
                 keep_highest: bool = True,
                 restrict_n_samples: int = -1):
        self.fraction = fraction
        self.pred_results_file = Path(pred_results_file)
        self.keep_highest = keep_highest
        pred_results = _load_dict(self.pred_results_file)
        self.pruning_metric = pred_results['train']['entropies']
        assert isinstance(self.pruning_metric,
                          np.ndarray), f"pruning_metric must be a numpy array, but is {type(self.pruning_metric)}"
        self.restrict_n_samples = restrict_n_samples

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        assert len(dataset) == len(
            self.pruning_metric
        ), f"len(dataset) ({len(dataset)}) must be equal to len(pruning_metric) ({len(self.pruning_metric)})"
        sample_idxes = _get_sample_idxes_by_metric(self.pruning_metric, self.fraction, self.keep_highest)
        sample_idxes = get_random_subset_idxes(sample_idxes, self.restrict_n_samples)
        return data.Subset(dataset, sample_idxes)


_sample_selector_registry = {
    'random': RandomSampleSelector,
    'prediction_depth': PredictionDepthSampleSelector,
}


def get_sample_selector_class(sample_selector_name: str) -> Type[Callable]:
    if sample_selector_name in _sample_selector_registry:
        return _sample_selector_registry[sample_selector_name]
    else:
        assert False, f"Unknown sample selector name \"{sample_selector_name}\". Available sample selectors are: {str(_sample_selector_registry.keys())}"
