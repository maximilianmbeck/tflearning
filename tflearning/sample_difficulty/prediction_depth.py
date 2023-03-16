from pathlib import Path
import torch
import wandb
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from torch import nn
from torch.utils import data
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt

from .prediction_depth_script import PredictionDepthConfigInternal
"""Implementation of the prediction depth according to [1].

[1] Baldock, Robert J. N., Hartmut Maennel, and Behnam Neyshabur. 2021.
“Deep Learning Through the Lens of Example Difficulty.”
arXiv. https://doi.org/10.48550/arXiv.2106.09647.

"""

LOGGER = logging.getLogger(__name__)


class LayerFeatureExtractor(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 layer_names: List[str] = [],
                 features_before: bool = True,
                 append_softmax_output: bool = True):
        super().__init__()
        self.model = model
        self.features_before = features_before
        self.append_softmax_output = append_softmax_output
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

        # register save_features_hook for selected layers
        for name, module in self.model.named_modules():
            if name in layer_names:
                module.register_forward_hook(save_features_hook)
                self.layer_names_ordered.append(name)
        if self.append_softmax_output:
            self.layer_names_ordered.append("softmax_output")

    def get_ordered_layer_names(self, layer_names: List[str], append_softmax_output: bool = True):
        """Return an from input to output ordered list of layer names."""
        layer_names_ordered = []
        for name, module in self.model.named_modules():
            if name in layer_names:
                layer_names_ordered.append(name)
        if append_softmax_output:
            layer_names_ordered.append("softmax_output")
        return layer_names_ordered

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            self.features = []
            x = self.model(x)
            feature_dict = {layer: feature for layer, feature in zip(self.layer_names_ordered, self.features)}
            softmax_output = F.softmax(x, dim=1)
            if self.append_softmax_output:
                feature_dict["softmax_output"] = softmax_output.detach()

            msg = f"Expected {len(self.layer_names_ordered)} features, got {len(self.features)}"
            assert len(feature_dict) == len(self.layer_names_ordered), msg

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
                 train_dataset: data.Dataset,
                 config: PredictionDepthConfigInternal,
                 val_dataset: data.Dataset = None,
                 **kwargs):

        self.config = config

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = self.config.device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # update model batchnorm statistics
        self.model.to(device=self.device)
        self.model.eval()

        if self.config.update_bn_stats:
            LOGGER.info('Updating batchnorm statistics')
            torch.optim.swa_utils.update_bn(data.DataLoader(train_dataset, batch_size=self.config.batch_size),
                                            self.model,
                                            device=self.device)

        # get kNN train set indices
        self.knn_train_indices = np.random.default_rng().choice(len(train_dataset),
                                                                self.config.knn_n_train_samples,
                                                                replace=False)

        self.num_classes = None
        self.layer_names_ordered = None
        self.results = None

    def _extract_layer_features(self,
                                dataloader: data.DataLoader,
                                layer_name: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features from the model for the given layer. If no layer is given, the last layer is used.
        The dataloader must NOT shuffle the data."""
        feature_batches = []
        label_batches = []
        model_prediction_batches = []
        if layer_name is None:
            feature_extractor = LayerFeatureExtractor(self.model, [],
                                                      self.config.features_before,
                                                      append_softmax_output=True)
        else:
            feature_extractor = LayerFeatureExtractor(self.model, [layer_name],
                                                      self.config.features_before,
                                                      append_softmax_output=False)

        feature_extractor.eval()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            pred, feature_dict = feature_extractor(x)
            assert len(feature_dict) == 1, "Only one layer can be extracted at a time."
            # move the tensors to the cpu and covert them to numpy arrays
            # this avoids running out of memory on the cuda device
            for key, value in feature_dict.items():
                feature_batches.append(value.cpu().numpy())
            label_batches.append(y.cpu().numpy())
            model_prediction_batches.append(pred.detach().cpu().numpy())

        # concatenate the batches
        feats = np.concatenate(feature_batches, axis=0)
        layer_features = feats.reshape(feats.shape[0], -1)

        labels = np.concatenate(label_batches, axis=0)
        model_predictions = np.concatenate(model_prediction_batches, axis=0).argmax(axis=-1)
        if self.num_classes is None:
            self.num_classes = labels.max() + 1
        if self.layer_names_ordered is None:
            self.layer_names_ordered = feature_extractor.layer_names_ordered

        return layer_features, labels, model_predictions

    def _predict_layer_knn(self, layer_features: np.ndarray,
                           gt_labels: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        assert layer_features.shape[0] == gt_labels.shape[0], "Number of features and labels must match."
        # compute kNN predictions
        knn_classifier = KNeighborsClassifier(n_neighbors=self.config.knn_n_neighbors, **self.config.knn_kwargs)
        # we use the true labels
        knn_classifier.fit(layer_features[self.knn_train_indices], gt_labels[self.knn_train_indices])
        knn_predictions = knn_classifier.predict(layer_features)
        return knn_predictions

    def _compute_layer_accuracies(self, kNN_predictions: Dict[str, np.ndarray], labels: np.ndarray,
                                  final_predictions: np.ndarray) -> Dict[str, float]:

        layer_accuracies = {}
        for layer_name, layer_predictions in kNN_predictions.items():
            layer_accuracies[layer_name] = np.mean(layer_predictions == labels)
        layer_accuracies["model_preds"] = np.mean(final_predictions == labels)
        return layer_accuracies

    def _compute_prediction_depths(self,
                                   kNN_predictions: Dict[str, np.ndarray],
                                   labels: np.ndarray,
                                   final_predictions: np.ndarray,
                                   mode: str = 'model_prediction') -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Compute prediction depths for each sample.

        Args:
            kNN_predictions (Dict[str, np.ndarray]): _description_
            labels (np.ndarray): _description_
            final_predictions (np.ndarray): _description_
            mode (str): model_prediction: use the models prediction
                        for assigning the prediction depth
                        ground_truth_label: use the true label
                        for assigning the prediction depth

        Returns:
            np.ndarray: the prediction depth for each sample
        """
        modes = ['model_prediction', 'ground_truth_label', 'last_layer_knn_prediction']
        assert mode in modes, f"Unknown mode {mode}."

        if mode == 'model_prediction':
            compare_labels = final_predictions
        elif mode == 'ground_truth_label':
            compare_labels = labels
        elif mode == 'last_layer_knn_prediction':
            compare_labels = kNN_predictions[self.layer_names_ordered[-1]]
        else:
            raise ValueError(f"Unknown mode {mode}")

        ground_truth_labels = labels

        # compute the prediction depth for each sample
        # dim: (n_samples, n_layers)
        layer_preds = np.stack([kNN_predictions[k] for k in kNN_predictions.keys()], axis=1)

        pred_depths_correct, pred_depths_wrong = pred_depth_fn(layer_preds, compare_labels, ground_truth_labels)
        pred_depths = {'correct': pred_depths_correct, 'wrong': pred_depths_wrong}
        return pred_depths, layer_preds

    def _compute_for_dataset(self, dataset: data.Dataset) -> Dict[str, Union[Dict[str, float], np.ndarray]]:

        # get the ordered layer names
        layer_feature_extractor = LayerFeatureExtractor(self.model)
        self.layer_names_ordered = layer_feature_extractor.get_ordered_layer_names(self.config.layer_names,
                                                                                   self.config.append_softmax_output)

        # get the dataloader
        dataloader = data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)

        knn_layer_preds = {}
        # for each layer extract features and predict kNN
        pbar = tqdm(self.layer_names_ordered)
        for layer_name in pbar:
            pbar.set_description(f"kNN Layer {layer_name}")
            layer_features, labels, preds = self._extract_layer_features(dataloader=dataloader, layer_name=layer_name)
            knn_preds = self._predict_layer_knn(layer_features, labels)
            knn_layer_preds[layer_name] = knn_preds

        layer_accs = self._compute_layer_accuracies(knn_layer_preds, labels, preds)
        pred_depths, layer_preds = self._compute_prediction_depths(knn_layer_preds,
                                                                   labels,
                                                                   preds,
                                                                   mode=self.config.prediction_depth_mode)

        ret_dict = {'layer_accs': layer_accs, 'pred_depths': pred_depths, 'layer_preds': layer_preds, 'labels': labels}
        return ret_dict

    def compute(self) -> Tuple[Dict[str, float], np.ndarray]:
        """Compute the layer accuracies and the prediction
        depths for the train and val dataset.

        Returns:
            Tuple[Dict[str, float], np.ndarray]: layer accuracies and prediction depths
        """
        ret_dict = {}
        LOGGER.info("Computing layer accuracies and prediction depths for train dataset")
        ret_dict['train'] = self._compute_for_dataset(self.train_dataset)
        if self.val_dataset is not None:
            LOGGER.info("Computing layer accuracies and prediction depths for val dataset")
            ret_dict['val'] = self._compute_for_dataset(self.val_dataset)
        self.results = ret_dict
        return ret_dict

    def make_plots(self) -> List[plt.Figure]:
        """Plot the layer accuracies and prediction
        depths for the train and val dataset."""
        self.results = self.compute()
        return self._make_plots(self.results, save_dir=self.save_dir)

    def _make_plots(self,
                    pred_depth_results: Dict[str, Dict[str, Any]],
                    save_format: str = 'png',
                    save_dir: Union[Path, str] = './') -> List[plt.Figure]:
        figures = []
        for dataset, res_dict in pred_depth_results.items():
            LOGGER.info(f'Plotting dataset: {dataset}')
            f, axes = plt.subplots(1, 3, figsize=(3 * 12 * 1 / 2.54, 1.5 * 8 * 1 / 2.54))
            f.suptitle(f"Layer accs + prediction depths for {dataset} dataset-{self.experiment_specifier}", y=1.05)
            axes.flatten().tolist()
            self._plot_accuracies(axes[0], res_dict['layer_accs'])
            self._plot_prediction_depths_hist(axes[1:], res_dict['pred_depths'], dataset)
            figures.append(f)
            if save_format:
                f.savefig(f'{str(save_dir)}/pred_depth-{self.experiment_specifier}-dataset_{dataset}.{save_format}',
                          dpi=300,
                          bbox_inches='tight')
        return figures

    def _plot_accuracies(self, ax, layer_accs: Dict[str, float]) -> None:
        model_pred_acc = layer_accs.pop('model_preds', -1)
        layer_accs_vals = np.array(list(layer_accs.values()))
        layer_ind = np.arange(len(layer_accs_vals))
        ax.plot(layer_ind, layer_accs_vals)
        ax.grid(True)
        ax.set_ylim(0.0, 1.0)
        ax.set_title('kNN layer accs')
        if self.wandb_run is not None:
            tbl_data = [[layer, acc] for layer, acc in zip(layer_ind, layer_accs_vals)]
            tbl = wandb.Table(data=tbl_data, columns=['layer', 'acc'])
            wandb.log({f'kNN layer accs': wandb.plot.line(tbl, x='layer', y='acc', title='kNN layer accs')})

    def _plot_prediction_depths_hist(self, axes, pred_depths: Dict[str, np.ndarray], dataset: str = '') -> None:

        def plot_hist(ax, pred_depths, title):
            bins = np.arange(0, len(self.layer_names_ordered) + 1, 1) - 0.5
            ax.hist(pred_depths, bins=bins)
            x_labels = self.layer_names_ordered.copy()
            ax.set_xticks(ticks=np.arange(0, len(self.layer_names_ordered), 1), labels=x_labels, rotation=90)
            ax.set_xlim(-1.0, len(self.layer_names_ordered))
            ax.grid(True)
            ax.set_title(title)
            if self.wandb_run is not None:
                hist, bin_edges = np.histogram(pred_depths, bins=bins)
                sample_count = hist.tolist()
                pred_depth = np.arange(0, len(self.layer_names_ordered) + 1, 1).tolist()
                tbl_data = [[pd, sc] for pd, sc in zip(pred_depth, sample_count)]
                tbl = wandb.Table(data=tbl_data, columns=['pred_depth', 'sample_count'])
                wandb.log(
                    {f'{title}': wandb.plot.bar(table=tbl, label='pred_depth', value='sample_count', title=title)})

        for ax, (mode, pred_depths) in zip(axes, pred_depths.items()):
            plot_hist(ax, pred_depths, f'[{mode}] pred depths\n{self.prediction_depth_mode}')
            wandb.log({f'{dataset}-{mode} samples': np.logical_not(np.isnan(pred_depths)).sum()})
            LOGGER.info(f'{mode} predicted samples: {np.logical_not(np.isnan(pred_depths)).sum()}')


# if too slow use numba
def pred_depth_fn(preds, pred_depth_labels, ground_truth_labels):
    """Differentiates between correct and wrong
    predictions and computes the prediction depth."""

    pred_depths_correct = np.zeros(preds.shape[0])
    pred_depths_wrong = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        for j in reversed(range(preds.shape[1])):
            if pred_depth_labels[i] == ground_truth_labels[i]:
                # model prediction is correct
                if preds[i, j] != pred_depth_labels[i]:
                    pred_depths_correct[i] = j
                    break
                pred_depths_wrong[i] = float('nan')
            else:
                # model prediction is wrong
                if preds[i, j] != pred_depth_labels[i]:
                    pred_depths_wrong[i] = j
                    break
                pred_depths_correct[i] = float('nan')
    return pred_depths_correct, pred_depths_wrong


# if too slow use numba
def pred_depth_fn_simple(preds, labels):
    """This sets prediction depth to -1 if the prediction
    is wrong for last layer (i.e. 'label' here)."""
    pred_depths = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        for j in reversed(range(preds.shape[1])):
            if j == preds.shape[1] - 1:
                if preds[i, j] != labels[i]:
                    pred_depths[i] = -1
                    break
            if preds[i, j] != labels[i]:
                pred_depths[i] = j
                break
    return pred_depths
