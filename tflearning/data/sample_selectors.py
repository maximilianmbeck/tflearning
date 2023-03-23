import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Type

import numpy as np
import pandas as pd
import torch
from torch.utils import data

LOGGER = logging.getLogger(__name__)


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

def count_samples_per_class(dataset: data.Dataset) -> Dict[int, int]:
    """Count samples per class in a dataset."""
    class_counts = {}
    for sample in dataset:
        label = sample[1]
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    return class_counts

def get_random_subset_idxes(indices: np.ndarray, restrict_n_samples: int = -1) -> np.ndarray:
    """Take an iid subset of the indices."""
    assert isinstance(indices, np.ndarray), f"indices must be a numpy array, but is {type(indices)}"
    if restrict_n_samples > 0:
        assert restrict_n_samples <= len(
            indices
        ), f"Not enough samples in the datset: restrict_n_samples ({restrict_n_samples}) must be smaller than len(indices) ({len(indices)}). "
        indices = indices[torch.randperm(len(indices))[:restrict_n_samples]]
    return indices


def get_class_sorted_entropies(entropies: np.ndarray, class_idxes: np.ndarray) -> Dict[int, pd.Series]:
    """Get entropies split by class idxes.
    """
    assert len(entropies) == len(
        class_idxes
    ), f'entropies and class_labels must have same length, but are {len(entropies)} and {len(class_idxes)}'
    cf_labels_entropy = np.stack([entropies, class_idxes], axis=1)
    entr_labels_df = pd.DataFrame(cf_labels_entropy, columns=['entropy', 'label'])
    entropies_per_label_df = entr_labels_df.pivot(columns='label', values='entropy')
    class_sorted_entropy_arrays = {}
    for c in entropies_per_label_df.columns:
        class_idx = int(c)
        class_sorted_entropy_arrays[class_idx] = entropies_per_label_df[~entropies_per_label_df[c].isnull()][c]
    return class_sorted_entropy_arrays


class RandomSampleSelector:

    def __init__(self, fraction: float, restrict_n_samples: int = -1):
        self.fraction = fraction
        self.restrict_n_samples = restrict_n_samples

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        n_samples = len(dataset)
        n_samples_to_select = int(n_samples * self.fraction)
        indices = torch.randperm(n_samples)[:n_samples_to_select].numpy()
        indices = get_random_subset_idxes(indices, self.restrict_n_samples)
        return data.Subset(dataset, indices)


class PruningMetricSampleSelector:

    def __init__(self,
                 fraction: float,
                 pruning_metric_file: str,
                 pruning_metric: str, 
                 keep_highest: bool = True,
                 restrict_n_samples: int = -1):
        self.fraction = fraction
        self.pruning_metric_file = Path(pruning_metric_file)
        self.keep_highest = keep_highest
        pred_results = _load_dict(self.pruning_metric_file)
        pruning_metric_extractor = get_pruning_metric_extractor(pruning_metric)
        self.pruning_metric, _ = pruning_metric_extractor(pred_results)
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

class PruningMetricClassBalanceSampleSelector:

    def __init__(self,
                 fraction: float,
                 pruning_metric_file: str,
                 pruning_metric: str,
                 keep_highest: bool = True,
                 restrict_n_samples: int = -1):
        self.fraction = fraction
        self.pruning_metric_file = Path(pruning_metric_file)
        self.keep_highest = keep_highest
        pred_results = _load_dict(self.pruning_metric_file)
        pruning_metric_extractor = get_pruning_metric_extractor(pruning_metric)
        self.pruning_metric, self.labels = pruning_metric_extractor(pred_results)
        assert isinstance(self.pruning_metric,
                          np.ndarray), f"pruning_metric must be a numpy array, but is {type(self.pruning_metric)}"
        assert isinstance(self.labels, np.ndarray), f"labels must be a numpy array, but is {type(self.labels)}"
        self.restrict_n_samples = restrict_n_samples

    def __call__(self, dataset: data.Dataset) -> data.Dataset:
        assert len(dataset) == len(
            self.pruning_metric
        ), f"len(dataset) ({len(dataset)}) must be equal to len(pruning_metric) ({len(self.pruning_metric)})"

        cls_sorted_entropies = get_class_sorted_entropies(self.pruning_metric, self.labels)
        samples_per_class = int(self.fraction * len(dataset) / len(cls_sorted_entropies))

        num_samples_added = 0
        sample_idxes = []
        sort_ascending = not self.keep_highest
        for cls, entropies in cls_sorted_entropies.items():
            cls_sample_idxes = entropies.sort_values(ascending=sort_ascending).index[:samples_per_class]
            sample_idxes.extend(cls_sample_idxes)
            num_samples_added += len(cls_sample_idxes)
        if num_samples_added < (samples_per_class * len(cls_sorted_entropies)):
            LOGGER.warning(
                f"Added {num_samples_added} samples to the subset, but should actually add {samples_per_class * len(cls_sorted_entropies)} samples. (fraction={self.fraction}, restrict_n_samples={self.restrict_n_samples}, samples_per_class={samples_per_class}, num_classes={len(cls_sorted_entropies)})"
            )
        else:
            LOGGER.info(
                f"Added {num_samples_added} samples to the subset. (fraction={self.fraction}, restrict_n_samples={self.restrict_n_samples}"
            )
        sample_idxes = np.array(sample_idxes)
        sample_idxes = get_random_subset_idxes(sample_idxes, self.restrict_n_samples)
        return data.Subset(dataset, sample_idxes)


def _extract_prediction_depth(metric_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return metric_dict['train']['entropies'], metric_dict['train']['labels']

def _extract_el2n(metric_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return metric_dict['el2n']['average'], metric_dict['labels']

def _extract_grand(metric_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return metric_dict['grand']['average'], metric_dict['labels']

_pruning_metric_extractor_registry = {
    'prediction_depth': _extract_prediction_depth,
    'el2n': _extract_el2n,
    'grand': _extract_grand,
}

_sample_selector_registry = {
    'random': RandomSampleSelector,
    'prediction_depth': PruningMetricSampleSelector,
    'prediction_depth_class_balance': PruningMetricClassBalanceSampleSelector,
}


def get_sample_selector_class(sample_selector_name: str) -> Type[Callable]:
    if sample_selector_name in _sample_selector_registry:
        return _sample_selector_registry[sample_selector_name]
    else:
        assert False, f"Unknown sample selector name \"{sample_selector_name}\". Available sample selectors are: {str(_sample_selector_registry.keys())}"

def get_pruning_metric_extractor(metric_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if metric_name in _pruning_metric_extractor_registry:
        return _pruning_metric_extractor_registry[metric_name]
    else:
        assert False, f"Unknown metric name \"{metric_name}\". Available metrics are: {str(_pruning_metric_extractor_registry.keys())}"