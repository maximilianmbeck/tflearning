import torch
import numpy as np
from typing import Dict, List, Tuple
from torch import nn
from torch.utils import data

import torch.nn.functional as F
"""Implementation of the prediction depth according to [1].

[1] Baldock, Robert J. N., Hartmut Maennel, and Behnam Neyshabur. 2021. “Deep Learning Through the Lens of Example Difficulty.” arXiv. https://doi.org/10.48550/arXiv.2106.09647.

"""

# take the representation of before the activation
# for resnet18 and 20 this is


class LayerFeatureExtractor(nn.Module):

    def __init__(self, model: nn.Module, layer_names: List[str], features_before: bool = True):
        super().__init__()
        self.model = model
        self.features_before = features_before

        self.features = []
        self.layer_names_ordered = []

        def save_features_hook(module, input, output):
            if self.features_before:
                features = input
            else:
                features = output
            if isinstance(features, tuple):
                features = features[0]
            features = features.detach()
            self.features.append(features)

        for name, module in self.model.named_modules():
            if name in layer_names:
                module.register_forward_hook(save_features_hook)
                self.layer_names_ordered.append(name)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.features = []
        x = self.model(x)
        assert len(self.features) == len(
            self.layer_names_ordered), f"Expected {len(self.layers)} features, got {len(self.features)}"
        feature_dict = {layer: feature for layer, feature in zip(self.layer_names_ordered, self.features)}
        softmax_output = F.softmax(x, dim=1)
        feature_dict["softmax_output"] = softmax_output.detach()
        return x, feature_dict


def find_layer_names(model: nn.Module, name_pattern: str) -> List[str]:
    layers = []
    for name, module in model.named_modules():
        if name_pattern in name:
            layers.append(name)
    return layers


class PredictionDepth:

    def __init__(self,
                 model: nn.Module,
                 layer_names: List[str],
                 dataloader: data.DataLoader,
                 features_before: bool = True,
                 device: torch.device = None):

        self.model = model
        self.layer_names = layer_names
        self.features_before = features_before
        self.feature_extractor = LayerFeatureExtractor(model, layer_names, features_before)
        self.dataloader = dataloader
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.feature_extractor.to(device)

        # stores the features for each batch
        self.layer_features = {}  # type: List[Dict[str, torch.Tensor]]
        self.labels = []  # type: List[torch.Tensor]
        self.predictions = []  # type: List[torch.Tensor]

    def _extract_features(self):
        feature_batches = []
        label_batches = []
        prediction_batches = []
        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)
            pred, feature_dict = self.feature_extractor(x)

            # move the tensors to the cpu and covert them to numpy arrays
            # this avoids running out of memory on the cuda device
            for key, value in feature_dict.items():
                feature_dict[key] = value.cpu().numpy()
            feature_batches.append(feature_dict)
            label_batches.append(y.cpu().numpy())
            prediction_batches.append(pred.detach().cpu().numpy())

        # concatenate the batches
        layer_names = list(feature_batches[0].keys())
        layer_features = {} # TODO from here
        for layer_name in layer_names:
            pass
        self.feature_batches = {
            key: np.concatenate([batch[key] for batch in layer_feature_batches
                                ], axis=0) for feature_batch in self.feature_batches
        }
        self.label_batches = np.concatenate(self.label_batches, axis=0)
        self.prediction_batches = np.concatenate(self.prediction_batches, axis=0)

