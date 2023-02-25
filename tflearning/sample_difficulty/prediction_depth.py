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
"""Implementation of the prediction depth according to [1].

[1] Baldock, Robert J. N., Hartmut Maennel, and Behnam Neyshabur. 2021. “Deep Learning Through the Lens of Example Difficulty.” arXiv. https://doi.org/10.48550/arXiv.2106.09647.

"""

LOGGER = logging.getLogger(__name__)


class LayerFeatureExtractor(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 layer_names: List[str],
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

        for name, module in self.model.named_modules():
            if name in layer_names:
                module.register_forward_hook(save_features_hook)
                self.layer_names_ordered.append(name)
        if self.append_softmax_output:
            self.layer_names_ordered.append("softmax_output")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.features = []
        x = self.model(x)
        feature_dict = {layer: feature for layer, feature in zip(self.layer_names_ordered, self.features)}
        softmax_output = F.softmax(x, dim=1)
        if self.append_softmax_output:
            feature_dict["softmax_output"] = softmax_output.detach()
        assert len(feature_dict) == len(
            self.layer_names_ordered), f"Expected {len(self.layer_names_ordered)} features, got {len(self.features)}"
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
                 train_dataloader: data.DataLoader,
                 val_dataloader: data.DataLoader = None,
                 experiment_specifier: str = '',
                 save_dir: Path = './',
                 knn_n_neighbors: int = 30,
                 knn_kwargs: Dict[str, Any] = {'n_jobs': 10},
                 prediction_depth_mode: str = 'last_layer_knn_prediction',
                 features_before: bool = True,
                 append_softmax_output: bool = True,
                 device: torch.device = None,
                 wandb_run=None,
                 **kwargs):

        self.model = model
        self.layer_names = layer_names
        self.features_before = features_before
        self.feature_extractor = LayerFeatureExtractor(model,
                                                       layer_names,
                                                       features_before,
                                                       append_softmax_output=append_softmax_output)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.feature_extractor.to(device)

        self.knn_n_neighbors = knn_n_neighbors
        self.knn_kwargs = knn_kwargs
        self.prediction_depth_mode = prediction_depth_mode

        self.experiment_specifier = experiment_specifier
        self.save_dir = Path(save_dir)
        self.wandb_run = wandb_run

        self.num_classes = None
        self.layer_names_ordered = None
        self.results = None

    def _extract_features(self, dataloader: data.DataLoader) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        feature_batches = []
        label_batches = []
        prediction_batches = []
        for x, y in dataloader:
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
        layer_features = {}
        for layer_name in layer_names:
            feats = np.concatenate([batch[layer_name] for batch in feature_batches], axis=0)
            feats = feats.reshape(feats.shape[0], -1)
            layer_features[layer_name] = feats

        labels = np.concatenate(label_batches, axis=0)
        predictions = np.concatenate(prediction_batches, axis=0).argmax(axis=-1)
        if self.num_classes is None:
            self.num_classes = labels.max() + 1
        if self.layer_names_ordered is None:
            self.layer_names_ordered = self.feature_extractor.layer_names_ordered

        return layer_features, labels, predictions

    def _predict_layer_knn(self,
                           val_dataloader: data.DataLoader) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        #! this needs a lot of memory as the full train and the full val set (including the hidden representations) are stored in memory
        # extract features train dataset
        LOGGER.info("Extracting features for train and val dataset")
        train_features, train_labels, train_predictions = self._extract_features(self.train_dataloader)
        # extract features
        val_features, val_labels, val_predictions = self._extract_features(val_dataloader)

        kNN_predictions = {}
        for layer_name, layer_val_features in tqdm(val_features.items(), desc="Computing kNN predictions"):
            layer_train_features = train_features[layer_name]
            # compute kNN predictions
            knn_classifier = KNeighborsClassifier(n_neighbors=self.knn_n_neighbors, **self.knn_kwargs)
            # we use the true labels
            knn_classifier.fit(layer_train_features, train_labels)
            kNN_predictions[layer_name] = knn_classifier.predict(layer_val_features)

        # output: Dict[str, np.ndarray] containing knn predictions per layer
        # output: np.ndarray true labes
        # output: np.ndarray predicted labels
        return kNN_predictions, val_labels, val_predictions

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
            mode (str): model_prediction: use the models prediction for assigning the prediction depth
                        ground_truth_label: use the true label for assigning the prediction depth

        Returns:
            np.ndarray: the prediction depth for each sample
        """
        assert mode in ['model_prediction', 'ground_truth_label', 'last_layer_knn_prediction'], f"Unknown mode {mode}."

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

    def _compute_for_dataloader(self, dataloader: data.DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        knn_preds, labels, preds = self._predict_layer_knn(dataloader)
        layer_accs = self._compute_layer_accuracies(knn_preds, labels, preds)
        pred_depths, layer_preds = self._compute_prediction_depths(knn_preds,
                                                                   labels,
                                                                   preds,
                                                                   mode=self.prediction_depth_mode)

        return layer_accs, pred_depths, layer_preds

    def compute(self) -> Tuple[Dict[str, float], np.ndarray]:
        """Compute the layer accuracies and the prediction depths for the train and val dataset.

        Returns:
            Tuple[Dict[str, float], np.ndarray]: layer accuracies and prediction depths
        """
        ret_dict = {}
        LOGGER.info("Computing layer accuracies and prediction depths for train dataset")
        train_layer_accs, train_pred_depths, train_layer_preds = self._compute_for_dataloader(self.train_dataloader)
        ret_dict['train'] = {
            'layer_accs': train_layer_accs,
            'pred_depths': train_pred_depths,
            'layer_preds': train_layer_preds
        }
        if self.val_dataloader is not None:
            LOGGER.info("Computing layer accuracies and prediction depths for val dataset")
            val_layer_accs, val_pred_depths, val_layer_preds = self._compute_for_dataloader(self.val_dataloader)
            ret_dict['val'] = {
                'layer_accs': val_layer_accs,
                'pred_depths': val_pred_depths,
                'layer_preds': val_layer_preds
            }
        self.results = ret_dict
        return ret_dict

    def make_plots(self) -> List[plt.Figure]:
        """Plot the layer accuracies and prediction depths for the train and val dataset."""
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

    def _plot_prediction_depths_hist(self, axes, pred_depths: Dict[str, np.ndarray], dataset: str='') -> None:

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
    """Differentiates between correct and wrong predictions and computes the prediction depth."""
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
    """This sets prediction depth to -1 if the prediction is wrong for last layer (i.e. 'label' here)."""
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
